import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Wrapper class for either HpN or DDX7 or Wavetable
'''
class DDSP_Decoder(nn.Module):
    def __init__(self, decoder, synth):
        super().__init__()
        self.decoder = decoder
        self.synth = synth

    def forward(self,x):
        x = self.decoder(x)
        return self.synth(x)

    def get_sr(self):
        return self.synth.sample_rate

    def get_params(self):
        return self.decoder.output_keys

    def get_params_size(self):
        params_size = {}
        params_names = self.decoder.output_keys
        
        for i, name in enumerate(params_names):
            params_size[name] = self.decoder.output_sizes[i]
        return params_size
    
    
class DDSP_Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, synth):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.synth = synth
        
    def forward(self, x):
        x = self._encode(x)
        x = self.decoder(x)
        return self.synth(x)
    
    def _encode(self, x):
        z = self.encoder(x['audio'])
        x['z'] = z
        return x
    
    def get_sr(self):
        return self.synth.sample_rate

    def get_params(self):
        return self.decoder.output_keys

    def get_params_size(self):
        params_size = {}
        params_names = self.decoder.output_keys
        
        for i, name in enumerate(params_names):
            params_size[name] = self.decoder.output_sizes[i]
        return params_size
    