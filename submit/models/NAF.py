import torch
from torch import nn
from .PosEnc import PosEncoder
import math
import torch.nn.functional as F

class NAF(nn.Module):
    def __init__(self, grid_size=[256, 8, 16], layer_channels=[512, 512, 256], 
                 idx_dim=512, decoder_channels=[512, 512, 512], scene_x=5, scene_y=3,
                 class_values=[], class_bin_size=2, activation="None",
                 wave_length=32768):
        super(NAF, self).__init__()
        # assert grid_size[0] % 6 == 0
        self.wave_length = wave_length

        self.scene_x = scene_x
        self.scene_y = scene_y
        self.feature_grid = nn.Parameter(torch.rand(grid_size))

        # output sequence index encoder
        self.grid_coordinates = nn.Parameter(torch.arange(int(self.wave_length // 2)).unsqueeze(1), requires_grad=False)
        self.idx_encoder = PosEncoder(each_input_embed_size=idx_dim)

        # classification regression head
        self.class_values = nn.Parameter(torch.tensor(class_values).to(torch.float32), requires_grad=False)
        self.class_bin_size = class_bin_size

        # encoder (input is position embedding)
        self.layers = nn.Sequential()
        in_channels = grid_size[0] * 2
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
        
        if len(class_values) > 0:
            self.use_class = True
            self.real_head = nn.Linear(decoder_channels[-1], len(class_values) +1)
            self.imag_head = nn.Linear(decoder_channels[-1], len(class_values) +1)
        else:
            self.use_class = False
            self.real_head = nn.Linear(decoder_channels[-1], 1)
            self.imag_head = nn.Linear(decoder_channels[-1], 1)

        self.activation = activation
        
    def grid_parameters(self):
        return [self.feature_grid]
    
    def backbone_parameters(self):
        return self.layers.parameters()
    
    def head_parameters(self):
        modules_ = [self.decoder, self.real_head, self.imag_head]
        return sum([list(m.parameters()) for m in modules_], [])
    
    def class_regress_combination(self, class_out, reg_out):
        '''
        class_out shape: [B, class_num, spectrum_length]
        reg_out shape: [B, 1, spectrum_length]
        '''
        weights = torch.nn.functional.gumbel_softmax(class_out, dim=1, hard=True) # [B, class_num, spec_length]
        class_out = (self.class_values[None, :, None] * weights).sum(dim=1, keepdim=True) # [B, 1, spec_length]

        # reg_out = reg_out.clip(min=-self.class_bin_size / 2, max=self.class_bin_size / 2)
        reg_out = reg_out * 0.05
        reg_out = torch.tanh(reg_out) * self.class_bin_size / 2

        return class_out + reg_out

    def forward(self, input_posA, input_posB):
        '''
        interpolate to get corresponding feature in the grid, concat to form one feature as input to linear layer
        '''
        # 1. embed position using unlearnable encoder
        # skip

        # 2. extract corresponding position's grid feature
        scene_size = torch.tensor([self.scene_x, self.scene_y], device=input_posA.device)
        cond_coordinate = input_posA[..., :2] / scene_size * 2 - 1
        tgt_coordinate = input_posB[..., :2] / scene_size * 2 - 1

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
        
        x = torch.hstack([cond_feature, tgt_feature]).unsqueeze(1) # [B, 1, 2 * grid_size[0]]

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
        x = torch.cat([x.repeat(1, int(self.wave_length // 2), 1), freq_time_pos_emb], dim=-1) # [B, spec_length, idx_dim + channel]

        # 5. decoder and head inference
        for i, layer in enumerate(self.decoder):
            x_next = layer(x)
            if x_next.shape == x.shape:
                x = x_next + x # [B, spec_length // seg_size, channel]
            else:
                x = x_next # [B, spec_length // seg_size, channel]

        real = self.real_head(x).permute(0, 2, 1) # [B, class_num, spec_length]
        imag = self.imag_head(x).permute(0, 2, 1) # [B, class_num, spec_length]
        
        # 6. get numerical value from class prediction
        if self.use_class:
            real = self.class_regress_combination(class_out=real[:, :-1], reg_out=real[:, -1:])
            imag = self.class_regress_combination(class_out=imag[:, :-1], reg_out=imag[:, -1:]) # [B, 1, spec_length]

        if self.activation == "sigmoid":
            real = real * 0.05
            real = torch.sigmoid(real) * 12
            imag = imag * 0.05
            imag = torch.sigmoid(imag) * 12

        # 7. get complex warp result
        half_warp_fft = torch.complex(real, imag)[:, 0] # [B, spec_length]

        warp_fft = torch.hstack([half_warp_fft, torch.conj(half_warp_fft.flip((-1, )))])

        return warp_fft






