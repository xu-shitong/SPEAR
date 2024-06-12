import torch
import torch.nn as nn

class PosEncoder(nn.Module):
    def __init__(self, input_dim = 3,
                 append_input = False,
                 each_input_embed_size = 256,
                 encode_fns = [torch.sin, torch.cos]):
        super(PosEncoder, self).__init__()
        self.input_dim = input_dim
        self.append_input = append_input
        assert each_input_embed_size % 2 == 0
        self.each_input_embed_size = each_input_embed_size
        self.encode_fns = encode_fns
        self.build_embed_fns()

    def build_embed_fns(self):
        embed_fns = list()
        output_dim = 0

        if self.append_input:
            embed_fns.append(lambda x: x)
            output_dim += self.input_dim

        for embed_id in range(self.each_input_embed_size//2):
            freq_band = 1./10000.**((2*embed_id)/self.each_input_embed_size)
            for fn_tmp in self.encode_fns:
                embed_fns.append(lambda x, fn_tmp=fn_tmp, freq_band = freq_band: fn_tmp(x*freq_band))
                output_dim += self.input_dim

        self.embed_fns = embed_fns
        self.output_dim = output_dim

    def forward(self, input_pos):
        pos_encode = torch.cat([fn(input_pos) for fn in self.embed_fns], -1)

        return pos_encode