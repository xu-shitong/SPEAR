import torch
from data.TestDataset import TestDataset
from data.TrainDataset import TrainDataset
from torch.utils.data import DataLoader
import models.NAF as NAF
import models.SPEAR as SPEAR
from models.InterpolateBaseline import Interpolate_Model
import os
from tqdm import tqdm
from pesq import pesq_batch
import torch.nn.functional as F
from math import sqrt
import torchvision
from skimage.metrics import structural_similarity as ssim


GPU = "2"
DATASET = "grid-sample_full_sine_sweep-scale-2000-step-0.05_no-varyZ_test_data"
scale = 1

threshould = 3

# =========================== our model =============================
wave_length = 32768
LOAD_MODEL = "" #  our final model
model = SPEAR.SPEAR(
                    grid_size=[192, 8, 16],
                    seg_size=384,
                    layer_channels=[512, 512, 512, 384],
                    tf_layer_num=12,
                    scene_x=5, scene_y=3,
                    add_fix_pos=[False, False], refine_fix_pos=False,
                    wave_length=wave_length
                    )
model.load_state_dict(torch.load(
    LOAD_MODEL,
    map_location=torch.device('cpu'))["state_dict"],
    strict=True)

# # ========================== naf ==============================
# wave_length = 32768
# LOAD_MODEL = "" # naf final model
# model = NAF.NAF(grid_size = [256, 8, 16],
#                 layer_channels=[512, 512, 256], 
#                 decoder_channels=[512, 512, 512], 
#                 scene_x=5, scene_y=3,
#                 class_values=[], class_bin_size=2,
#                 activation=None, wave_length=wave_length
#                 )
# model.load_state_dict(torch.load(
#     LOAD_MODEL,
#     map_location=torch.device('cpu'))["state_dict"],
#     strict=True)


# # ========================= interpolate 5 ==============================
# wave_length = 32768
# LOAD_MODEL = "Interp 5 model"
# model = Interpolate_Model(TrainDataset("grid-sample_full_sine_sweep-scale-2000-step-0.05_no-varyZ_train_data", 
#                                      sample_size=3000, clip_warped_audio=False, sample_rate=16384, wave_length=32768, 
#                                      posB="fixed", snr_db=30, prep_kernel_size=0),
#                          k=5, threshould=threshould)

# # ========================= nearest ===============================
# wave_length = 32768
# LOAD_MODEL = "Nearest neighbour model"
# model = Interpolate_Model(TrainDataset("grid-sample_full_sine_sweep-scale-2000-step-0.05_no-varyZ_train_data", 
#                                      sample_size=3000, clip_warped_audio=False, sample_rate=16384, wave_length=32768, 
#                                      posB="fixed", snr_db=20, prep_kernel_size=0),
#                          k=1, threshould=threshould)

os.environ["CUDA_VISIBLE_DEVICES"] = GPU
torch.cuda.manual_seed(42)
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
print("using device: ", dev)

model.to(device)
model.eval()

dataset = TestDataset(DATASET, sample_size=10000, clip_warped_audio=False, sample_rate=16384, 
                          wave_length=wave_length, snr_db=0, threshould=threshould, scale=sqrt(scale))
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=4)


human_dataset1 = TestDataset("grid-sample_p257_001-scale-2000-step-0.05_no-varyZ_test_data", 
                                 sample_size=10000, clip_warped_audio=False, sample_rate=16384, snr_db=0, 
                                 prep_kernel_size=0, threshould=threshould, wave_length=wave_length)
human_dataset2 = TestDataset("grid-sample_p257_009-scale-2000-step-0.05_no-varyZ_test_data", 
                                 sample_size=10000, clip_warped_audio=False, sample_rate=16384, snr_db=0, 
                                 prep_kernel_size=0, threshould=threshould, wave_length=wave_length)
human_dataset3 = TestDataset("grid-sample_p257_261-scale-2000-step-0.05_no-varyZ_test_data", 
                                 sample_size=10000, clip_warped_audio=False, sample_rate=16384, snr_db=0, 
                                 prep_kernel_size=0, threshould=threshould, wave_length=wave_length)
human_dataset4 = TestDataset("grid-sample_p257_289-scale-2000-step-0.05_no-varyZ_test_data", 
                                 sample_size=10000, clip_warped_audio=False, sample_rate=16384, snr_db=0, 
                                 prep_kernel_size=0, threshould=threshould, wave_length=wave_length)
human_dataset5 = TestDataset("grid-sample_p257_347-scale-2000-step-0.05_no-varyZ_test_data", 
                                 sample_size=10000, clip_warped_audio=False, sample_rate=16384, snr_db=0, 
                                 prep_kernel_size=0, threshould=threshould, wave_length=wave_length)

def single_pra_metrics(model, dataloader):

    def calculate_psnr_warpfield(warpfield1, warpfield2):
        def scale_warpfield(wave):
            min_val = torch.min(wave, dim=-1, keepdim=True)[0] # shape: [B, 1]
            max_val = torch.max(wave, dim=-1, keepdim=True)[0]
            scale = 255 / (max_val - min_val)
            shift = 255 - scale * max_val

            scaled_warpfield = scale * wave + shift

            return scaled_warpfield
            
        warpfield1 = scale_warpfield(warpfield1)
        warpfield2 = scale_warpfield(warpfield2)

        mse = torch.mean((warpfield1 - warpfield2) ** 2, dim=-1)
        mse[mse == 0] = 100

        # return 10 * np.log10(max_value / (np.sqrt(mse)))
        return (10 * torch.log10(255**2 / mse)).mean().item()

    def compute_SDR(wave_pred, wave_gt):
        signal_amplitude = torch.sum(wave_gt ** 2, dim=-1)
        diff = torch.sum((wave_gt - wave_pred) ** 2, dim=-1)
        sdr = 10 * torch.log10(signal_amplitude / diff)

        return sdr.mean().item()

    def ssim_warpfield(wave_pred, wave_gt):
        def get_img(wave):
            wave_img = torch.stft(wave, 
                                    n_fft=1024, 
                                    hop_length=256, 
                                    window=torch.hann_window(1024).to(device),
                                    return_complex=True)
            
            real_scale = 255/(wave_img.real.amax(dim=(1, 2), keepdim=True) - wave_img.real.amin(dim=(1, 2), keepdim=True))
            real_offset = 255 - real_scale * wave_img.real.amax(dim=(1, 2), keepdim=True)
            
            imag_scale = 255/(wave_img.imag.amax(dim=(1, 2), keepdim=True) - wave_img.imag.amin(dim=(1, 2), keepdim=True))
            imag_offset = 255 - imag_scale * wave_img.imag.amax(dim=(1, 2), keepdim=True)

            return (wave_img.real * real_scale + real_offset).cpu().numpy(), (wave_img.imag * imag_scale + imag_offset).cpu().numpy()
        
        wave_pred_real_img, wave_pred_imag_img = get_img(wave_pred)
        wave_gt_real_img, wave_gt_imag_img = get_img(wave_gt)

        score_real = ssim(wave_pred_real_img, wave_gt_real_img, 
                          data_range=255, 
                          full=False, channel_axis=0)
        score_imag = ssim(wave_pred_imag_img, wave_gt_imag_img, 
                          data_range=255, 
                          full=False, channel_axis=0)
        return (score_real + score_imag) / 2



    psnr_real_acc = 0
    psnr_imag_acc = 0
    sdr_real_acc = 0
    sdr_imag_acc = 0
    t_mse_real_acc = 0
    t_mse_imag_acc = 0
    ssim_real_acc = 0
    ssim_imag_acc = 0

    warped_fft_real_mse = 0
    warped_fft_imag_mse = 0
    warped_wave_mse = 0
    for audio_cond, pos_cond, audio_tgt, pos_tgt, _, _ in tqdm(dataloader):
        audio_cond = audio_cond.to(device)
        pos_cond = pos_cond.to(device)
        audio_tgt = audio_tgt.to(device)
        pos_tgt = pos_tgt.to(device)
        
        with torch.no_grad():
            # predict
            warp_pred = model(pos_cond, pos_tgt)

            # get ground truth warp field
            fft1 = torch.fft.fft(audio_cond, norm="backward")
            fft2 = torch.fft.fft(audio_tgt, norm="backward")
            warp = (fft2 / fft1)[:, 0]
            warp.real = torch.nan_to_num(warp.real, nan=0)
            warp.imag = torch.nan_to_num(warp.imag, nan=0)
            warp.real = warp.real.clip(min=-threshould, max=threshould)
            warp.imag = warp.imag.clip(min=-threshould, max=threshould)

            # psnr
            psnr_real_acc += calculate_psnr_warpfield(warpfield1=warp_pred.real, warpfield2=warp.real)
            psnr_imag_acc += calculate_psnr_warpfield(warpfield1=warp_pred.imag, warpfield2=warp.imag)

            # sdr
            sdr_real_acc += compute_SDR(warp_pred.real, warp.real)
            sdr_imag_acc += compute_SDR(warp_pred.imag, warp.imag)

            # mse
            t_mse_real_acc += F.mse_loss(warp_pred.real, warp.real).item()
            t_mse_imag_acc += F.mse_loss(warp_pred.imag, warp.imag).item()

            # ssim
            ssim_real_acc += ssim_warpfield(warp_pred.real, warp.real)
            ssim_imag_acc += ssim_warpfield(warp_pred.imag, warp.imag)

            # warped audio fft mse loss
            fft2_ = warp_pred.unsqueeze(1) * fft1
            warped_fft_real_mse += F.mse_loss(fft2_.real, fft2.real).item()
            warped_fft_imag_mse += F.mse_loss(fft2_.imag, fft2.imag).item()

            # warped audio l2 loss
            wave2_pred = torch.fft.ifft(fft2_, norm="backward").real
            warped_wave_mse += F.mse_loss(wave2_pred, audio_tgt).item()

    psnr_avg = (psnr_real_acc + psnr_imag_acc) / 2 / len(dataloader)
    sdr_avg = (sdr_real_acc + sdr_imag_acc) / 2 / len(dataloader)
    t_mse_avg = (t_mse_real_acc + t_mse_imag_acc) / 2 / len(dataloader)
    fft_avg = (warped_fft_real_mse + warped_fft_imag_mse) / 2 / len(dataloader)
    ssim_avg = (ssim_real_acc + ssim_imag_acc) / 2 / len(dataloader)

    print("model: ", LOAD_MODEL)
    print("dataset: ", DATASET)
    print("psnr: ", psnr_avg)
    print("sdr: ", sdr_avg)
    print("ssim: ", ssim_avg)
    print("t_mse: ", t_mse_avg)
    print("warped fft mse: ", fft_avg)
    print("warped wave mse: ", warped_wave_mse / len(dataloader))

def human_voise_metrics(model, datasets):
    def get_pesq_score(gt_sound, pred_sound):
        sample_rate = 16000
        pesq_score = pesq_batch(sample_rate, gt_sound, pred_sound, 'wb', n_processor=4)
        return sum(pesq_score) / len(pesq_score)

    print("evaluating", LOAD_MODEL)
    pesq_acc = 0
    for human_dataset in datasets:
        dataloader = DataLoader(dataset=human_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=4)

        one_dataset_pseq = 0
        for audio_cond, pos_cond, audio_tgt, pos_tgt, _, _ in tqdm(dataloader):
            audio_cond = audio_cond.to(device)
            # audio_tgt = audio_tgt.to(device)
            pos_cond = pos_cond.to(device)
            pos_tgt = pos_tgt.to(device)

            audio_cond = audio_cond[:, 0]
            audio_tgt = audio_tgt[:, 0]

            with torch.no_grad():
                warp_pred = model(pos_cond, pos_tgt)

                fft1 = torch.fft.fft(audio_cond, norm="backward")
                audio_tgt_pred = torch.fft.ifft(warp_pred * fft1, norm="backward").real
                
                pesq_score = get_pesq_score(audio_tgt.numpy(), audio_tgt_pred.detach().cpu().numpy())
                # pesq_score = get_pesq_score(audio_tgt, audio_tgt_pred)

                one_dataset_pseq += pesq_score
        one_dataset_pseq = one_dataset_pseq / len(dataloader)
        pesq_acc += one_dataset_pseq
        print("finished one dataset pseq", one_dataset_pseq)
    print("pseq: ", pesq_acc / len(datasets))

single_pra_metrics(model, dataloader)

human_voise_metrics(model, [
                            human_dataset1, 
                            human_dataset2, 
                            human_dataset3, 
                            human_dataset4, 
                            human_dataset5
                            ])
