import torch
from data.R2RDataset import R2RDataset
from torch.utils.data import DataLoader
import models.NAF as NAF
import models.SPEAR as SPEAR
from Loss import *
import os
from tqdm import tqdm
from attrdict import AttrDict
import traceback
import yaml


# Import packages
import sys,humanize,psutil,GPUtil

# Define function
def mem_report():
    print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))

    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))

def get_config(yaml_config_filename):
    with open(yaml_config_filename) as f:
        config_dict = yaml.safe_load(f)

    return config_dict

def loss_func(args, warp, audio_cond, audio_tgt):
    '''
    warp: predicted fft of warp field 

    audio_cond: audio recorded at reference position
    audio_tgt: audio recorded at target position, inference target
    '''

    pooled_loss = pooled_supervision(depth=3, warp_pred=warp, audio_cond=audio_cond, audio_tgt=audio_tgt, regional_loss=args.regional_loss)

    supervise_warp_field_l = supervise_fft_warp_field(audio_cond=audio_cond, audio_tgt=audio_tgt, warp_pred=warp, regional_loss=args.regional_loss, 
                                                      scale_loss=args.scale_loss, pred_range=args.pred_range, wave_length=args.wave_length)

    warped_audio_l = supervise_output_audio(audio_cond=audio_cond, audio_tgt=audio_tgt, warp_pred=warp, regional_loss=args.regional_loss)

    warped_audio_waveform_l = supervise_output_audio_waveform(audio_cond, audio_tgt, warp, regional_loss=args.regional_loss)

    self_consistency_l = torch.zeros((1,), device=audio_cond.device)
    
    close_range_l = torch.zeros((1,), device=audio_cond.device)

    return pooled_loss, supervise_warp_field_l, warped_audio_l, warped_audio_waveform_l, self_consistency_l, close_range_l


def train_func(args, epoch, model, basis_model, dataloader, optimizer=None, device=torch.device("cpu"), train=True):
    if train:
        model.train()
    else:
        model.eval()

    acc_losses = [0] * 7
    
    titer = dataloader
    if train:
        titer = tqdm(dataloader, unit="iter")
    for i, (audio_cond, pos_cond, audio_tgt, pos_tgt) in enumerate(titer):
        audio_cond = audio_cond.to(device)
        pos_cond = pos_cond.to(device)
        audio_tgt = audio_tgt.to(device)
        pos_tgt = pos_tgt.to(device)

        warp = model(pos_cond, pos_tgt)
        if args.basis_model != "":
            with torch.no_grad():
                warp_basis = basis_model(pos_cond, pos_tgt)
            warp = warp + warp_basis

        losses = loss_func(args, warp, audio_cond, audio_tgt)
            
        pooled_loss, supervise_warp_field_l, warped_audio_l, warped_audio_waveform_l, self_consistency_l, close_range_l = losses

        l = args.PooledLossWeight * pooled_loss + args.SuperviseSpectrumLossWeight * supervise_warp_field_l \
            + args.SpectrumLossWeight * warped_audio_l + args.WarpedWaveformL2Weight * warped_audio_waveform_l \
            + args.SelfConsistencyWeight * self_consistency_l + args.CloseRangeWeight * close_range_l
        if train:
            titer.set_description(f"iter {i}")
            titer.set_postfix(loss=l.item(), 
                              warped_audio_l=warped_audio_l.item(),
                              sup_warp=supervise_warp_field_l.item(), 
                              waveform_l=warped_audio_waveform_l.item(),
                              self_consis=self_consistency_l.item(),
                              close_range_l=close_range_l.item()
                              )

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        acc_losses[1:] = [(acc_l + l).item() for acc_l, l in zip(acc_losses[1:], losses)]
        acc_losses[0] += l.item()

    return [l / len(dataloader) for l in acc_losses]


def main_func(args):

    mem_report()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print("using device: ", dev)

    process_id = os.getpid()
    print("Start experiment", process_id)
    log = open(f"{process_id}.txt", "a")
    log.write("\n".join([str(key) + " " + str(args.get(key)) for key in args.keys()]) + "\n") # write hyperparameter in log file

    # dataset dataloader
    train_dataset = R2RDataset(f"{args.dataset_tag}_train_data", sample_size=args.sample_size, clip_warped_audio=args.clip_warped_audio, 
                               sample_rate=args.sampling_rate, wave_length=args.wave_length, posB=args.posB, snr_db=args.snr_db, prep_kernel_size=args.prep_kernel_size)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=6)
    val_dataset = R2RDataset(f"{args.dataset_tag}_val_data", sample_size=args.val_sample_size, clip_warped_audio=args.clip_warped_audio, 
                             sample_rate=args.sampling_rate, wave_length=args.wave_length, posB=args.posB, snr_db=args.snr_db, prep_kernel_size=args.prep_kernel_size)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=6)

    # model
    if args.model_name == "naf":
        model = NAF.R2R_1DLinear(grid_size = args.grid_size,
                                  layer_channels=args.layer_channels, 
                                  decoder_channels=args.decoder_channels, 
                                  scene_x=args.scene_x, scene_y=args.scene_y,
                                  class_values=args.class_values, class_bin_size=args.class_bin_size,
                                  activation=args.activation, wave_length=args.wave_length
                                  )
        model.to(device)
    elif args.model_name == "spear":
        model = SPEAR.R2R_1DTF_3d_bert(grid_size = args.grid_size,
                                  seg_size=args.seg_size,
                                  layer_channels=args.layer_channels,
                                  tf_layer_num=args.tf_layer_num, 
                                  scene_x=args.scene_x, scene_y=args.scene_y,
                                  add_fix_pos=args.add_fix_pos, refine_fix_pos=args.refine_fix_pos, 
                                  wave_length=args.wave_length
                                  )
        model.to(device)
    else:
        raise NotImplementedError(args.model_name)
    
    if args.load_model != "":
        model.load_state_dict(torch.load(args.load_model)["state_dict"], strict=True)

    basis_model = None
    # optimizer
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            [
                dict(name='grid', params=model.grid_parameters(), lr=args.grid_lr),
                dict(name='backbone', params=model.backbone_parameters(), lr=args.backbone_lr),
                dict(name='head', params=model.head_parameters(), lr=args.head_lr),
            ], 
            lr=args.backbone_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay,
            amsgrad=True)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            [
                dict(name='grid', params=model.grid_parameters(), lr=args.grid_lr),
                dict(name='backbone', params=model.backbone_parameters(), lr=args.backbone_lr),
                dict(name='head', params=model.head_parameters(), lr=args.head_lr),
            ], 
            lr=args.backbone_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay,
            amsgrad=True)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            [
                dict(name='grid', params=model.grid_parameters(), lr=args.grid_lr),
                dict(name='backbone', params=model.backbone_parameters(), lr=args.backbone_lr),
                dict(name='head', params=model.head_parameters(), lr=args.head_lr),
            ], 
            lr=args.backbone_lr,)    
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=args.lr_decay_epoch,
                                                   gamma=args.lr_decay_gamma)

    # train
    for epoch in range(args.epoch_num):
        train_losses = train_func(args, epoch, model, basis_model, train_dataloader, optimizer, device=device, train=True)
        train_losses = [str(num) for num in train_losses]

        with torch.no_grad():
            val_losses = train_func(args, epoch, model, basis_model, val_dataloader, device=device, train=False)
            val_losses = [str(num) for num in val_losses]

        log.write(f"Epoch: {epoch}, train_losses: {', '.join(train_losses)}, " + 
                  f"val_losses: {', '.join(val_losses)}, lr:{lr_scheduler.get_last_lr()[0]}\n")
        print(f"Finish {epoch} / {args.epoch_num}")

        lr_scheduler.step()

        if epoch % args.save_epoch == 0 and epoch != 0:
            torch.save({"state_dict": model.state_dict()}, f"{process_id}_{epoch}.pt")
    torch.save({"state_dict": model.state_dict()}, f"{process_id}.pt")
    log.close()

    mem_report()
    print("Finish experiment", process_id)

if __name__ == "__main__":
    process_id = os.getpid()
    try:
        print("running ", process_id)

        args_dict = get_config(sys.argv[1])
        args = AttrDict(args_dict)

        main_func(args)
    except Exception:
        print("training failed", traceback.format_exc())

