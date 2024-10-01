import torch
from torch import nn
from .PosEnc import PosEncoder
import math
import torch.nn.functional as F
import numpy as np

def distance(x1, x2):
    # by jacobrgardner
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    return res

def fit_predict_torch(input_pos:torch.Tensor, input_target:torch.Tensor, predict_pos:torch.Tensor, bandwidth:torch.Tensor) -> torch.Tensor:
    dist_vector = -distance(predict_pos, input_pos)
    gauss_dist = torch.exp(dist_vector/(2.0 * torch.square(bandwidth.unsqueeze(0))))
    magnitude = torch.sum(gauss_dist, dim=1, keepdim=True)
    out = torch.mm(gauss_dist, input_target)/magnitude
    return out

class NAF(nn.Module):
    def __init__(self, grid_dim=64,
                 bandwidth_min=0.1, bandwidth_max=0.5,
                 float_amt=0.1, grid_bandwidth=0.25,
                 layer_channels=[512, 512, 512, 256], 
                 decoder_channels=[512, 512, 512, 512],
                 idx_dim=512, scene_x=5, scene_y=3):
        super(NAF, self).__init__()

        self.scene_x = scene_x
        self.scene_y = scene_y

        grid_coors_x = np.arange(0, scene_x, 0.25)
        grid_coors_y = np.arange(0, scene_y, 0.25)
        grid_coors_x, grid_coors_y = np.meshgrid(grid_coors_x, grid_coors_y)
        grid_coors_x = grid_coors_x.flatten()
        grid_coors_y = grid_coors_y.flatten()
        xy_train = np.array([grid_coors_x, grid_coors_y]).T
        self.bandwidth_min = bandwidth_min
        self.bandwidth_max = bandwidth_max
        self.float_amt = float_amt
        self.bandwidths = nn.Parameter(torch.zeros(len(grid_coors_x))+grid_bandwidth, requires_grad=True)
        self.register_buffer("grid_coors_xy",torch.from_numpy(xy_train).float(), persistent=True)
        self.xy_offset = nn.Parameter(torch.zeros_like(self.grid_coors_xy), requires_grad=True)
        self.grid_0 = nn.Parameter(torch.randn(len(grid_coors_x),grid_dim, device="cpu").float() / np.sqrt(float(grid_dim)), requires_grad=True)

        # output sequence index encoder
        self.grid_coordinates = nn.Parameter(torch.arange(16384).unsqueeze(-1), requires_grad=False)
        self.idx_encoder = PosEncoder(each_input_embed_size=idx_dim)

        # encoder (input is position embedding)
        self.layers = nn.Sequential()
        in_channels = grid_dim * 2
        for i, c in enumerate(layer_channels):
            self.layers.add_module(f"layer {i}", 
                nn.Sequential(
                    nn.Linear(in_channels, c),
                    nn.ReLU()))
            in_channels = c
        # decoder (input is encoder output + sequence index embedding)
        self.decoder = nn.Sequential()
        in_channels = layer_channels[-1] + idx_dim
        for i, c in enumerate(decoder_channels):
            self.decoder.add_module(f"layer {i}", 
                nn.Sequential(
                    nn.Linear(in_channels, c),
                    nn.ReLU()))
            in_channels = c

        # decoder (input is encoder output + sequence index embedding)
        self.real_head = nn.Linear(decoder_channels[-1], 1)
        self.imag_head = nn.Linear(decoder_channels[-1], 1)
        
    def grid_parameters(self):
        return []
    
    def backbone_parameters(self):
        return self.parameters()
    
    def head_parameters(self):
        return []
    
    def class_regress_combination(self, class_out, reg_out):
        '''
        class_out shape: [B, class_num, spectrum_length]
        reg_out shape: [B, 1, spectrum_length]
        '''
        weights = torch.nn.functional.gumbel_softmax(class_out, dim=1, hard=True) # [B, class_num, spec_length]
        class_out = (self.class_values[None, :, None] * weights).sum(dim=1, keepdim=True) # [B, 1, spec_length]

        reg_out = reg_out * 0.05
        reg_out = torch.tanh(reg_out) * self.class_bin_size / 2

        return class_out + reg_out

    def forward(self, input_posA, input_posB):
        '''
        interpolate to get corresponding feature in the grid, concat to form one feature as input to linear layer
        '''
        self.bandwidths.data = torch.clamp(self.bandwidths.data, self.bandwidth_min, self.bandwidth_max)

        grid_coors_baseline = self.grid_coors_xy + torch.tanh(self.xy_offset) * self.float_amt
        grid_feat_v0 = fit_predict_torch(grid_coors_baseline, self.grid_0, input_posA[..., :2], self.bandwidths)
        grid_feat_v1 = fit_predict_torch(grid_coors_baseline, self.grid_0, input_posB[..., :2], self.bandwidths)
        cond_feature = torch.cat((grid_feat_v0, grid_feat_v1), dim=-1)
                
        x = cond_feature.unsqueeze(1) # [B, 1, grid_size[0]]

        # 3. model inference
        for layer in self.layers:
            x_next = layer(x)
            if x_next.shape == x.shape:
                x = x_next + x # [B, 1, channel]
            else:
                x = x_next

        # 4. construct output sequence idx embedding
        freq_time_pos_emb = self.idx_encoder(self.grid_coordinates) # [spec_length, idx_dim]
        freq_time_pos_emb = freq_time_pos_emb.repeat(input_posA.shape[0], 1, 1) # [B, spec_length, idx_dim]

        ## add sequence idx embedding to encoded feature
        x = torch.cat([x.repeat(1, 16384, 1), freq_time_pos_emb], dim=-1) # [B, spec_length, idx_dim + channel]

        # 5. decoder and head inference
        for i, layer in enumerate(self.decoder):
            x_next = layer(x)
            if i == len(self.decoder) // 2:
                x = x_next + x
            else:
                x = x_next

        real = self.real_head(x) # [B, spec_length // seg_size, seg_size * (bin_size + 1)]
        imag = self.imag_head(x) # [B, spec_length // seg_size, seg_size * (bin_size + 1)]

        # 7. get complex rir result
        half_rir_fft = torch.complex(real, imag).reshape((input_posA.shape[0], -1)) # [B, spec_length]

        rir_fft = torch.hstack([half_rir_fft, torch.conj(half_rir_fft.flip((-1, )))])

        return rir_fft




