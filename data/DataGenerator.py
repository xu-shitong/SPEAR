import pyroomacoustics as pra
import os
import numpy as np
from scipy.io import wavfile
import librosa
import pickle
import soundfile
import random
import torch 
import sys

sigma2_awgn = None
step_size = 0.05
audio_name = "full_sine_sweep" # scale 2000
scale = 2000

if sys.argv[1] == "train":
    train = True
    data_dir = f"./grid-sample_{audio_name}-scale-{scale}-step-{step_size}_no-varyZ_train_data"
    random_seed = 1000
elif sys.argv[1] == "test":
    train = False
    data_dir = f"./grid-sample_{audio_name}-scale-{scale}-step-{step_size}_no-varyZ_test_data"
    random_seed = 43
else:
    raise NotImplementedError(sys.argv[1])

sample_rate = 16384

class SPEARDataGenerator():
    def __init__(self, random_seed = 100):
        np.random.seed(random_seed)

    def init_material(self, e_abs, e_scattering ):
        material = pra.Material(energy_absorption=e_abs,
                                scattering = e_scattering)

        return material

    def init_room(self):
        """Parameter Configuration"""
        room_Lx = 5.
        room_Ly = 3.
        room_Lz = 4
        use_ray_tracing = True
        fs = sample_rate
        e_abs = 0.3
        e_scatter = 0.15

        room_corner = np.array([[0, 0], [room_Lx, 0], [room_Lx, room_Ly], [0, room_Ly]]).T
        material = self.init_material(e_abs=e_abs,
                                      e_scattering=e_scatter)

        snr=random.normalvariate(mu=-10000, sigma=2)

        room = pra.Room.from_corners(
            corners=room_corner,
            absorption=None,
            fs=fs,
            t0=0.0,
            max_order=3,
            sigma2_awgn=sigma2_awgn,
            materials=material,
            ray_tracing=use_ray_tracing,
            air_absorption=True)

        room.extrude(room_Lz, materials=material)

        # Set the ray tracing parameters
        room.set_ray_tracing(receiver_radius=0.5,
                             n_rays=1000,
                             energy_thres=1e-5)

        return room, [room_Lx, room_Ly, room_Lz], [e_abs, e_scatter]

    def add_sound_sources(self, pra_room ):
        seed_sound_filename = f"res/{audio_name}.wav"

        audio_anechoic, sr = librosa.load(seed_sound_filename, sr=sample_rate)
        audio_anechoic = audio_anechoic[:sample_rate]

        ss_loc = np.array([2., 2., 2.], np.float32)

        pra_room.add_source(ss_loc, signal=audio_anechoic * scale)

        return pra_room, ss_loc

    def add_mono_microphones(self, pra_room):
        # Generate the grid of coordinates
        x_coords = np.arange(0.5, 4.5, step_size)
        y_coords = np.arange(0.5, 2.5, step_size)
        X, Y = np.meshgrid(x_coords, y_coords)

        # Flatten the grid coordinates for easier processing
        coords = np.vstack((X.ravel(), Y.ravel(), np.zeros_like(X.ravel()))).T
        np.random.shuffle(coords)


        # Function to check if two points are adjacent
        def is_adjacent(pt1, pt2, step_size):
            return abs(pt1[0] - pt2[0]) <= step_size + 1e-3 and abs(pt1[1] - pt2[1]) <= step_size + 1e-3

        # Select a subset of coordinates such that no two points are adjacent
        selected_coords = []
        not_selected_coords = []
        for coord in coords:
            if len(selected_coords) < 300 and not any(is_adjacent(coord, selected, step_size) for selected in selected_coords):
                selected_coords.append(coord)
            else:
                not_selected_coords.append(coord)

        # Convert selected_coords to numpy array for easier handling
        selected_coords = np.array(selected_coords)
        not_selected_coords = np.array(not_selected_coords)

        if train:
            mic_loc = not_selected_coords
        else:
            mic_loc = selected_coords

        for mic_loc_tmp in mic_loc:
            pra_room.add_microphone(loc=mic_loc_tmp, fs=pra_room.fs)


        return pra_room, mic_loc

    def simulate_sound(self, pra_room):
        # compute and pad a rir with zeros in the end
        pra_room.compute_rir()
        for m in range(len(pra_room.rir)):
            for s in range(len(pra_room.rir[0])):
                pad_length = sample_rate - pra_room.rir[m][s].shape[0]
                if pad_length <= 0:
                    continue
                pra_room.rir[m][s] = np.pad(pra_room.rir[m][s], (0, pad_length))
        pra_room.simulate()

        return pra_room.mic_array.signals, pra_room

    def simulate_dataset(self):
        pra_room, room_size, abs_coeff = self.init_room()
        pra_room, ss_loc = self.add_sound_sources(pra_room)
        pra_room, mic_loc = self.add_mono_microphones(pra_room)

        simulated_waveforms, pra_room = self.simulate_sound(pra_room=pra_room)
        # sample_rate = 24000

        torch.save(pra_room.rir, data_dir + "/rirs.pt")

        #save the simulated waveforms
        save_dir = data_dir
        room_param = dict()
        room_param['abs_coeff'] = abs_coeff
        room_param['ss_loc'] = ss_loc

        with open(os.path.join(save_dir,'room_param.pickle'), 'wb') as handle:
            pickle.dump(room_param, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for mic_id in range(mic_loc.shape[0]):
            mic_tmp = dict()
            mic_tmp['mic_waveform'] = simulated_waveforms[mic_id,:]
            mic_tmp['mic_loc'] = mic_loc[mic_id,:]

            mic_loc_tmp = mic_loc[ mic_id, : ]

            save_wave_basename = 'mic_{}_{:.4f}_{:.4f}_{:.4f}.pickle'.format(mic_id,
                                                                             mic_loc_tmp[0],
                                                                             mic_loc_tmp[1],
                                                                             mic_loc_tmp[2])

            soundfile.write(os.path.join(save_dir, save_wave_basename.replace('.pickle','.wav')),
                            # mic_tmp['mic_waveform'].astype(np.int16)[0:sample_rate],
                            mic_tmp['mic_waveform'].astype(np.int16),
                            samplerate=sample_rate,
                            subtype='PCM_16')

os.makedirs(data_dir, exist_ok=True)
spearGen = SPEARDataGenerator(random_seed=random_seed)

spearGen.simulate_dataset()












