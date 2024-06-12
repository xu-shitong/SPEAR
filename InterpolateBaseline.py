import torch
from torch import nn
import math
import re
import os
from torch.utils.data import DataLoader


class Interpolate_Model(nn.Module):
    def __init__(self, dataset, k, threshould):
        super(Interpolate_Model, self).__init__()

        dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=4)

        positions = torch.zeros((0, 3))
        audios = torch.zeros((0, 32768))
        for audio_cond, pos_cond, _, _, _, _ in dataloader:
            positions = torch.vstack([positions, pos_cond])
            audios = torch.vstack([audios, audio_cond[:, 0]])

        self.positions = nn.Parameter(positions) # [audio_num, 3]
        self.audios = nn.Parameter(audios) # [audio_num, 32768]

        self.k = k

        self.threshould = threshould
    
    def get_k_nearest_audio(self, pos):
        pos = pos.unsqueeze(1)
        # pos shape: [B, 1, 3]
        distances = torch.norm(self.positions - pos, dim=-1) # [B, audio_num]

        # Get the indices of the k smallest distances
        _, indices = torch.topk(distances, self.k, largest=False, dim=-1) # [B, k]

        # Select the corresponding rows from the values tensor
        k_nearest_values = self.audios[indices] # [B, k, 32768]

        return k_nearest_values


    def forward(self, input_posA, input_posB):
        posA_audios = self.get_k_nearest_audio(input_posA) # [B, k, 32768]
        posB_audios = self.get_k_nearest_audio(input_posB)

        posA_fft = torch.fft.fft(posA_audios, norm="backward").unsqueeze(2) # [B, k, 1, 32768]
        posB_fft = torch.fft.fft(posB_audios, norm="backward").unsqueeze(1) # [B, 1, k, 32768]
        warp = (posB_fft / posA_fft) # [B, k, k, 32768]
        warp = torch.nan_to_num(warp, nan=0)
        warp.real = warp.real.clip(min=-self.threshould, max=self.threshould)
        warp.imag = warp.imag.clip(min=-self.threshould, max=self.threshould)

        warp = warp.mean(dim=[1, 2]) # [B, 32768]

        return warp








