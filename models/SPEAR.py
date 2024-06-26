import torch
from torch import nn
from .PosEnc import PosEncoder
import math
import torch.nn.functional as F
import math


class SPEAR(nn.Module):
    def __init__(self, grid_size=[192, 8, 16], seg_size=384, layer_channels=[512, 512, 512, 384], 
                 tf_layer_num=12, scene_x=5, scene_y=3,
                 add_fix_pos=[False, False], refine_fix_pos=False, pred_range=16384, wave_length=32768):
        super(SPEAR, self).__init__()
        assert layer_channels[-1] == seg_size
        assert layer_channels[-1] == grid_size[0] * 2

        self.wave_length = wave_length

        self.pred_range = pred_range

        # learnable grid feature
        self.scene_x = scene_x
        self.scene_y = scene_y
        self.feature_grid = nn.Parameter(torch.rand(grid_size))

        # unlearned position encoder
        self.add_fix_pos = add_fix_pos
        if refine_fix_pos:
            self.fix_pos_projection = nn.Linear(2 * grid_size[0], 2 * grid_size[0])
        else:
            self.fix_pos_projection = nn.Identity()
        self.fix_pos_encoder = PosEncoder(each_input_embed_size=grid_size[0] // 3)

        self.grid_coordinates = nn.Parameter(torch.arange(math.ceil(pred_range / seg_size)).unsqueeze(1), requires_grad=False)
        self.fix_idx_encoder = PosEncoder(each_input_embed_size=seg_size)

        # encoder (input is position embedding)
        self.layers = nn.Sequential()
        in_channels = grid_size[0] * 2
        for i, c in enumerate(layer_channels):
            self.layers.add_module(f"layer {i}", 
                nn.Sequential(
                    nn.Linear(in_channels, c),
                    nn.ReLU()))
            in_channels = c

        # transformer decoder
        _decoder_layer = nn.TransformerEncoderLayer(
            d_model=seg_size, 
            nhead=8, 
            dim_feedforward=2048,
            dropout=0.1,
            activation=F.relu,
            layer_norm_eps=0.00001,
            batch_first=True,
            norm_first=False,
            bias=True
        )
        self.decoder = nn.TransformerEncoder(_decoder_layer, num_layers=tf_layer_num)

        self.real_head = nn.Linear(seg_size, seg_size)
        self.imag_head = nn.Linear(seg_size, seg_size)
        
    def grid_parameters(self):
        return [self.feature_grid]
    
    def backbone_parameters(self):
        return list(self.layers.parameters()) + list(self.decoder.parameters()) + list(self.fix_pos_projection.parameters())
    
    def head_parameters(self):
        modules_ = [self.real_head, self.imag_head]
        return sum([list(m.parameters()) for m in modules_], [])
    
    def forward(self, input_posA, input_posB):
        '''
        interpolate to get corresponding feature in the grid, concat to form one feature as input to linear layer
        '''

        # 1. extract corresponding position's grid feature
        scene_size = torch.tensor([self.scene_x, self.scene_y], device=input_posA.device)
        cond_coordinate = input_posA[..., :2] / scene_size * 2 - 1
        tgt_coordinate = input_posB[..., :2] / scene_size * 2 - 1

        assert (cond_coordinate.abs() <= 1).all(), f"{cond_coordinate.min(dim=0)} {cond_coordinate.max(dim=0)}"
        assert (tgt_coordinate.abs() <= 1).all(), f"{tgt_coordinate.min(dim=0)} {tgt_coordinate.max(dim=0)}"

        ## cond_feature shape [B, grid_size[0]]
        cond_feature = torch.nn.functional.grid_sample(
            self.feature_grid.repeat(cond_coordinate.shape[0], 1, 1, 1), # [B, grid_size[0], grid_size[1], grid_size[2]]
            cond_coordinate[:, None, None, :], # [B, 1, 1, 2]
            align_corners=True).squeeze(dim=[2, 3])
        
        ## tgt_feature shape [B, grid_size[0]]
        tgt_feature = torch.nn.functional.grid_sample(
            self.feature_grid.repeat(tgt_coordinate.shape[0], 1, 1, 1), # [B, grid_size[0], grid_size[1], grid_size[2]]
            tgt_coordinate[:, None, None, :], # [B, 1, 1, 2]
            align_corners=True).squeeze(dim=[2, 3])

        x = torch.hstack([cond_feature, tgt_feature]) # [B, 2 * grid_size[0]]

        fixed_pos_embedding = self.fix_pos_projection(self.fix_pos_encoder(torch.hstack([input_posA, input_posB])))

        if self.add_fix_pos[0]:
            x = x + fixed_pos_embedding # [B, 2 * grid_size[0]]

        x = x.unsqueeze(1) # [B, 1, 2 * grid_size[0]]

        # 2. model inference, get refined position embedding of mic positions
        for layer in self.layers:
            x_next = layer(x)
            if x_next.shape == x.shape:
                x = x_next + x # [B, 1, channel]
            else:
                x = x_next

        # 3. get embedding representing each output segment index
        freq_time_pos_emb = self.fix_idx_encoder(self.grid_coordinates) # [spec_length // seg_size, seg_size]
        freq_time_pos_emb = freq_time_pos_emb.repeat(input_posA.shape[0], 1, 1) # [B, spec_length // seg_size, seg_size]
        
        x = x + freq_time_pos_emb # [B, spec_length // seg_size, seg_size]
        for decoder_layer in self.decoder.layers:
            x = decoder_layer(src=x)
            if self.add_fix_pos[1]:
                x = x + fixed_pos_embedding.unsqueeze(1)

        real = self.real_head(x) # [B, spec_length // seg_size, seg_size * (bin_size + 1)]
        imag = self.imag_head(x) # [B, spec_length // seg_size, seg_size * (bin_size + 1)]

        # 5. get complex warp result
        half_warp_fft = torch.complex(real, imag).reshape((input_posA.shape[0], -1)) # [B, spec_length]

        half_warp_fft = half_warp_fft[..., :self.pred_range]

        half_warp_fft = torch.nn.functional.pad(half_warp_fft, (0, int(self.wave_length // 2) - self.pred_range), mode="constant", value=0)

        warp_fft = torch.hstack([half_warp_fft, torch.conj(half_warp_fft.flip((-1, )))])

        return warp_fft