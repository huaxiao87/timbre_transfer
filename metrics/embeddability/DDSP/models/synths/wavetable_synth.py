"""
Wavetable Synthesizer. 
The code mainly comes from https://github.com/gudgud96/diff-wave-synth with minor adaptations.
"""

import torch
import torch.nn as nn
import numpy as np
import random
from .synth_utils import upsample, exp_sigmoid, amp_to_impulse_response, fft_convolve, Reverb



class WTpNsynth(nn.Module):
    ''' Wavetable + Noise Synthesizer (with Reverb) '''
    def __init__(self, sample_rate, block_size, reverb_scale = 1, scale_fn = exp_sigmoid):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.reverb = Reverb(length=int(sample_rate/reverb_scale), sample_rate=sample_rate)
        self.scale_fn = scale_fn
        
        self.wavetables = Wavetablesynth(sr=sample_rate, block_size=block_size)

    # expects: harmonic_distr, amplitude, noise_bands
    def forward(self, controls):

        attentions = self.scale_fn(controls['wave_attention'])
        noise_bands = self.scale_fn(controls['noise_bands'])
        total_amp = self.scale_fn(controls['amplitude'])

        # harmonics = remove_above_nyquist(
        #     harmonics,
        #     controls['f0_hz'],
        #     self.sample_rate,
        # )
        
        # Normalize the attentions
        attentions /= attentions.sum(-1, keepdim=True)

        attentions_up = upsample(attentions, self.block_size)
        f0_up = upsample(controls['f0_hz'], self.block_size,'linear')
        total_amp_up = upsample(total_amp, self.block_size)

        wavetable_signal = self.wavetables(f0_up, total_amp_up, attentions_up)
        impulse = amp_to_impulse_response(noise_bands, self.block_size)

        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
            ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        dereverb_signal = wavetable_signal + noise

        # Reverb part
        synth_signal = self.reverb(dereverb_signal)
        synth_out = {
            'synth_audio': synth_signal,
            'dereverb_audio' : dereverb_signal,
            'noise_audio' : noise,
            'wavetable_audio' : wavetable_signal,
            'amplitude' : controls['amplitude'],
            'wave_attention': controls['wave_attention'],
            'noise_bands': controls['noise_bands'],
            'f0_hz': controls['f0_hz']
            }

        return synth_out


class Wavetablesynth(nn.Module):
    def __init__(self,
                 wavetables=None,
                 n_wavetables=20,
                 wavetable_len=512,
                 block_size=64,
                 sr=44100):
        super(Wavetablesynth, self).__init__()
        if wavetables is None: 
            self.wavetables = []
            for _ in range(n_wavetables):
                cur = nn.Parameter(torch.empty(wavetable_len).normal_(mean=0, std=0.01))
                self.wavetables.append(cur)

            self.wavetables = nn.ParameterList(self.wavetables)

            for idx, wt in enumerate(self.wavetables):
                # following the paper, initialize f0-f3 wavetables and disable backprop
                if idx == 0:
                    wt.data = generate_wavetable(wavetable_len, np.sin, cycle=1, phase=random.uniform(0, 1))
                    wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = False
                elif idx == 1:
                    wt.data = generate_wavetable(wavetable_len, np.sin, cycle=2, phase=random.uniform(0, 1))
                    wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = False
                elif idx == 2:
                    wt.data = generate_wavetable(wavetable_len, np.sin, cycle=3, phase=random.uniform(0, 1))
                    wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = False
                elif idx == 3:
                    wt.data = generate_wavetable(wavetable_len, np.sin, cycle=4, phase=random.uniform(0, 1))
                    wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = False
                else:
                    wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = True
            
        else:
            self.wavetables = wavetables
            
        # Sample rate for wavetable synthesis
        self.sr = sr
        
    def copy_wavetables(self, source_wavetables, requires_grad=False):
        with torch.no_grad():
            for idx, wt in enumerate(self.wavetables):
                wt.data.copy_(source_wavetables[idx].data)
                if idx not in [0, 1, 2, 3]:
                    wt.requires_grad = requires_grad
                else:
                    wt.requires_grad = False

    def forward(self, pitch, amplitude, attention): 
               
        output_waveform_lst = []
        for wt_idx in range(len(self.wavetables)):
            wt = self.wavetables[wt_idx]
            if wt_idx not in [0, 1, 2, 3]:
                wt = nn.Tanh()(wt)  # ensure wavetable range is between [-1, 1]
            waveform = wavetable_osc(wt, pitch, self.sr)
            output_waveform_lst.append(waveform)

        # Stack the waveforms
        output_waveform = torch.stack(output_waveform_lst, dim=2)

        # Apply attention
        output_waveform = output_waveform * attention
        output_waveform_after = torch.sum(output_waveform, dim=2)
      
        # Apply amplitude
        output_waveform_after = output_waveform_after.unsqueeze(-1)
        output_waveform_after = output_waveform_after * amplitude
       
        return output_waveform_after
    
   

def wavetable_osc(wavetable, freq, sr):
    """
    General wavetable synthesis oscillator.
    """
    freq = freq.squeeze(-1) # remove the last dimension
    increment = freq / sr * wavetable.shape[0] # increment per sample
    index = torch.cumsum(increment, dim=-1) - increment[:,0].unsqueeze(-1) # start from index 0
    index = index % wavetable.shape[0] 

    # uses linear interpolation implementation
    index_low = torch.floor(index.clone())
    index_high = torch.ceil(index.clone())
    alpha = index - index_low
    index_low = index_low.long()
    index_high = index_high.long()
    
    assert index_low.shape == index_high.shape, "Index shape mismatch (low, high): {} vs {}".format(index_low.shape, index_high.shape)
    assert index_low.shape == alpha.shape, "Alpha shape mismatch: {} vs {}".format(index_low.shape, alpha.shape)
    assert torch.all(index_low >= 0) and torch.all(index_low < wavetable.shape[0]), "Index low out of bounds: {}. Should be [0, {}]".format(index_low, wavetable.shape[0])
    assert torch.all(index_high % wavetable.shape[0] >= 0) and torch.all(index_high % wavetable.shape[0] < wavetable.shape[0]), "Index low out of bounds: {}. Should be [0, {}]".format(index_low, wavetable.shape[0])
    assert torch.all(alpha >= 0) and torch.all(alpha < 1), "Alpha out of bounds: {}. Should be [0, 1]".format(alpha)

    output = wavetable[index_low] + alpha * (wavetable[index_high % wavetable.shape[0]] - wavetable[index_low])
        
    return output


def generate_wavetable(length, f, cycle=1, phase=0):
    """
    Generate a wavetable of specified length using 
    function f(x) where x is phase.
    Period of f is assumed to be 2 pi.
    """
    wavetable = np.zeros((length,), dtype=np.float32)
    for i in range(length):
        wavetable[i] = f(cycle * 2 * np.pi * i / length + 2 * phase * np.pi)
    return torch.tensor(wavetable)