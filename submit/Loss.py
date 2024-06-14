import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def spectral_convergence_loss(gt_fft, pred_fft):
    x_mag = torch.abs(pred_fft)
    y_mag = torch.abs(gt_fft)

    return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

def log_magnitude_loss(gt_fft, pred_fft):
    x_mag = torch.abs(pred_fft) + 1e-7
    y_mag = torch.abs(gt_fft) + 1e-7

    return F.l1_loss(torch.log(y_mag), torch.log(x_mag))

def real_imag_l1_loss(gt_fft, pred_fft):
    l1_loss = F.l1_loss(pred_fft.real, gt_fft.real) + F.l1_loss(pred_fft.imag, gt_fft.imag)
    return l1_loss

def real_imag_l2_loss(gt_fft, pred_fft):
    l2_loss = F.mse_loss(pred_fft.real, gt_fft.real) + F.mse_loss(pred_fft.imag, gt_fft.imag)
    return l2_loss


def supervise_fft_warp_field(audio_cond, audio_tgt, warp_pred, regional_loss=False, scale_loss=False, pred_range=16384, wave_length=32768):
    '''
    Calculate l1 and l2 loss between predicted and ground truth fft warp field
    '''
    if regional_loss:
        mask = torch.zeros_like(warp_pred.real, device=warp_pred.device, dtype=torch.bool, requires_grad=False)
        mask[..., warp_pred.shape[-1] // 3 : 2 * warp_pred.shape[-1] // 3] = True
        warp_pred = warp_pred.masked_fill(mask, 0)

    fft1 = torch.fft.fft(audio_cond, norm="backward")
    fft2 = torch.fft.fft(audio_tgt, norm="backward")
    warp = (fft2 / fft1)[:, 0]
    warp = torch.nan_to_num(warp, nan=0)
    warp.real = warp.real.clip(min=-10, max=10)
    warp.imag = warp.imag.clip(min=-10, max=10)

    if scale_loss:
        ratio = torch.linspace(1, 5, pred_range, device=warp.device, requires_grad=False)
        ratio = torch.nn.functional.pad(ratio, (0, int(wave_length // 2) - pred_range), mode="constant", value=0)
        ratio = torch.hstack([ratio, ratio.flip((-1, ))])
        ratio = ratio.repeat((warp.shape[0], 1))

        warp = warp * ratio
        warp_pred = warp_pred * ratio

    return real_imag_l1_loss(warp, warp_pred) + real_imag_l2_loss(warp, warp_pred)

def supervise_output_audio(audio_cond, audio_tgt, warp_pred, regional_loss=False):
    '''
    Calculate l1 and l2 loss between predicted and ground truth warped audio fft
    '''
    if regional_loss:
        mask = torch.zeros_like(warp_pred.real, device=warp_pred.device, dtype=torch.bool, requires_grad=False)
        mask[..., warp_pred.shape[-1] // 3 : 2 * warp_pred.shape[-1] // 3] = True
        warp_pred = warp_pred.masked_fill(mask, 0)

    fft1 = torch.fft.fft(audio_cond, norm="backward")[:, 0]
    fft2 = torch.fft.fft(audio_tgt, norm="backward")[:, 0]
    fft2_ = fft1 * warp_pred

    return spectral_convergence_loss(fft2, fft2_) + log_magnitude_loss(fft2, fft2_)

def supervise_output_audio_waveform(audio_cond, audio_tgt, warp_pred, regional_loss=False):
    '''
    Calculate l1 and l2 loss between predicted and ground truth warped audio waveform
    '''
    if regional_loss:
        mask = torch.zeros_like(warp_pred.real, device=warp_pred.device, dtype=torch.bool, requires_grad=False)
        mask[..., warp_pred.shape[-1] // 3 : 2 * warp_pred.shape[-1] // 3] = True
        warp_pred = warp_pred.masked_fill(mask, 0)

    fft1 = torch.fft.fft(audio_cond, norm="backward")[:, 0]
    fft2_ = fft1 * warp_pred
    audio_B = torch.fft.ifft(fft2_, norm="backward")

    return F.mse_loss(audio_B.real, audio_tgt[:, 0], reduction="sum")

def pooled_supervision(depth, warp_pred, audio_cond, audio_tgt, l1=True, l2=True, regional_loss=False):
    '''
    Calculate l1 and l2 loss between predicted and ground truth warped audio fft
    Use average pooling to get different resolution of warp field fft, calculate l1 l2 loss in each resolution 
    '''
    if regional_loss:
        mask = torch.zeros_like(warp_pred.real, device=warp_pred.device, dtype=torch.bool, requires_grad=False)
        mask[..., warp_pred.shape[-1] // 3 : 2 * warp_pred.shape[-1] // 3] = True
        warp_pred = warp_pred.masked_fill(mask, 0)

    fft1 = torch.fft.fft(audio_cond, norm="backward")
    fft2 = torch.fft.fft(audio_tgt, norm="backward")
    warp = (fft2 / fft1)[:, 0]
    warp = torch.nan_to_num(warp, nan=0)
    warp.real = warp.real.clip(min=-10, max=10)
    warp.imag = warp.imag.clip(min=-10, max=10)

    warp_pred = warp_pred.unsqueeze(1)
    pred_real = warp_pred.real
    pred_imag = warp_pred.imag

    warp = warp.unsqueeze(1)
    gt_real = warp.real
    gt_imag = warp.imag

    loss = 0
    for _ in range(depth):
        pred_real = torch.nn.functional.avg_pool1d(pred_real, kernel_size=3, stride=3, padding=0)
        pred_imag = torch.nn.functional.avg_pool1d(pred_imag, kernel_size=3, stride=3, padding=0)
        gt_real = torch.nn.functional.avg_pool1d(gt_real, kernel_size=3, stride=3, padding=0)
        gt_imag = torch.nn.functional.avg_pool1d(gt_imag, kernel_size=3, stride=3, padding=0)

        l1_loss, l2_loss = 0, 0
        if l2:
            l2_loss = (gt_real - pred_real).pow(2).mean() + (gt_imag - pred_imag).pow(2).mean()
        if l1:
            l1_loss = (gt_real - pred_real).abs().mean() + (gt_imag - pred_imag).abs().mean()

        loss = loss + l2_loss + l1_loss

    return loss


