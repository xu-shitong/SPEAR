import os
import torch
import torchaudio
import torchaudio.transforms as transforms
from torch.utils.data import Dataset
import re
import random
        
class TrainDataset(Dataset):
    def __init__(self, root_dir, sample_size, clip_warped_audio=False, sample_rate=20001, 
                 wave_length=20000, posB="fixed", snr_db=0, prep_kernel_size=0):
        super(TrainDataset, self).__init__()
        self.n_fft = 2048
        self.clip_warped_audio = clip_warped_audio
        self.sample_rate = sample_rate
        self.posB = posB
        self.wave_length = wave_length
        self.root_dir = root_dir
        self.audio_files = [f for f in os.listdir(root_dir) if f.endswith('.wav')]
        self.audio_files.sort()
        self.audio_files = self.audio_files[:sample_size]
        self.snr_db = snr_db

        self.prep_kernel_size = prep_kernel_size

        self.noise = torch.randn((len(self.audio_files), self.wave_length))

    def __len__(self):
        return len(self.audio_files)
    
    def pad_or_cut_wavefrom(self, wavefrom):
        # make sure all wavefroms are 1 seconds, i.e. length of sampling rate
        current_length = wavefrom.shape[-1]
        desired_length = self.wave_length
        if current_length < desired_length:
            padded_tensor = torch.nn.functional.pad(wavefrom, (0, desired_length - current_length), mode="constant", value=0)
            return padded_tensor
        cut_tensor = wavefrom[..., :desired_length]
        return cut_tensor

    def extract_position(self, s):
        # get mic coordinate from audio file name
        pattern = r'[-+]?\d*\.\d+|\d+'
        numbers = re.findall(pattern, s)
        numbers = [float(num) if '.' in num else int(num) for num in numbers]

        return numbers

    def normalize_warp_field(self, waveform1, waveform2):
        fft1 = torch.fft.fft(waveform1, norm="backward")
        fft2 = torch.fft.fft(waveform2, norm="backward")
        warp = (fft2 / fft1)
        warp = torch.nan_to_num(warp, nan=0)
        warp.real = warp.real.clip(min=-10, max=10)
        warp.imag = warp.imag.clip(min=-10, max=10)
        fft2_ = fft1 * warp

        waveform2_ = torch.fft.ifft(fft2_, norm="backward")
        return waveform2_.real
    
    def avg_pooling_preprocess(self, waveform1, waveform2):
        fft1 = torch.fft.fft(waveform1, norm="backward")
        fft2 = torch.fft.fft(waveform2, norm="backward")
        warp = (fft2 / fft1)
        warp = torch.nan_to_num(warp, nan=0)
        warp.real = warp.real.clip(min=-10, max=10)
        warp.imag = warp.imag.clip(min=-10, max=10)

        warp.real = torch.nn.functional.avg_pool1d(warp.real.unsqueeze(0), 
                                            kernel_size=self.prep_kernel_size, 
                                            stride=1, 
                                            padding=self.prep_kernel_size // 2
                                            ).squeeze()
        warp.imag = torch.nn.functional.avg_pool1d(warp.imag.unsqueeze(0), 
                                            kernel_size=self.prep_kernel_size, 
                                            stride=1, 
                                            padding=self.prep_kernel_size // 2
                                            ).squeeze()

        fft2_ = fft1 * warp

        waveform2_ = torch.fft.ifft(fft2_, norm="backward")
        return waveform2_.real

    
    def add_noise(self, wave, idx):
        if self.snr_db == 0:
            return wave

        snr = 10 ** (self.snr_db / 10)
        audio_power = torch.mean(wave**2)
        ratio = torch.sqrt(audio_power / snr)
        wave += self.noise[idx] * ratio

        return wave

    def __getitem__(self, idx):
        # get audio indexed by idx
        file_name1 = self.audio_files[idx]
        audio_file1 = os.path.join(self.root_dir, file_name1)
        waveform1, sr1 = torchaudio.load(audio_file1)
        assert sr1 == self.sample_rate, sr1
        waveform1 = self.pad_or_cut_wavefrom(waveform1)
        waveform1 = self.add_noise(waveform1, idx)

        ## randomly select another mic position as target mic
        ## model generate warp field to predict target mic position's sound
        if self.posB == "random":
            idx2 = random.randint(0, len(self.audio_files) - 1)
        elif self.posB == "fixed":
            idx2 = 1
        elif self.posB == "next":
            idx2 = (idx + 1) % len(self.audio_files) 
        file_name2 = self.audio_files[idx2]
        audio_file2 = os.path.join(self.root_dir, file_name2)
        waveform2, sr2 = torchaudio.load(audio_file2)
        assert sr2 == self.sample_rate, sr2
        waveform2 = self.pad_or_cut_wavefrom(waveform2)
        waveform2 = self.add_noise(waveform2, idx2)

        if self.prep_kernel_size != 0:
            waveform2 = self.avg_pooling_preprocess(waveform1, waveform2)

        if self.clip_warped_audio:
            waveform2 = self.normalize_warp_field(waveform1, waveform2)

        # extract position coordinates for 2 mics
        numbers1 = self.extract_position(file_name1)
        numbers2 = self.extract_position(file_name2)

        return waveform1, torch.tensor(numbers1[1:]), waveform2, torch.tensor(numbers2[1:])