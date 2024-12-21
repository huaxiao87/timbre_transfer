"""
Harmonic-plus-Noise Synthesizer. 
The code mainly comes from https://github.com/acids-ircam/ddsp_pytorch with minor adaptations.
"""

import torch
import torch.nn as nn
import math
from src.models.synths.synth_utils import remove_above_nyquist, upsample, amp_to_impulse_response, fft_convolve, exp_sigmoid, Reverb


class HpNsynth(nn.Module):
    ''' Harmonic + Noise Synthesizer (with Reverb) '''
    def __init__(self, sample_rate, block_size, reverb_scale = 1, scale_fn = exp_sigmoid):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.reverb = Reverb(length=int(sample_rate/reverb_scale), sample_rate=sample_rate)
        self.scale_fn = scale_fn

    # expects: harmonic_distr, amplitude, noise_bands
    def forward(self, controls):

        harmonics = self.scale_fn(controls['harmonic_distribution'])
        noise_bands = self.scale_fn(controls['noise_bands'])
        total_amp = self.scale_fn(controls['amplitude'])

        harmonics = remove_above_nyquist(
            harmonics,
            controls['f0_hz'],
            self.sample_rate,
        )
        
        # Normalize the harmonics
        harmonics /= harmonics.sum(-1, keepdim=True)
        # Scale the harmonics
        harmonics *= total_amp

        harmonics_up = upsample(harmonics, self.block_size)
        f0_up = upsample(controls['f0_hz'], self.block_size,'linear')

        harmonic_signal = harmonic_synth(f0_up, harmonics_up, self.sample_rate)
        impulse = amp_to_impulse_response(noise_bands, self.block_size)

        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
            ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        dereverb_signal = harmonic_signal + noise

        # Reverb part
        synth_signal = self.reverb(dereverb_signal)
        synth_out = {
            'synth_audio': synth_signal,
            'dereverb_audio' : dereverb_signal,
            'noise_audio' : noise,
            'harmonic_audio' : harmonic_signal,
            'amplitude' : controls['amplitude'],
            'harmonic_distribution': controls['harmonic_distribution'],
            'noise_bands': controls['noise_bands'],
            'f0_hz': controls['f0_hz']
            }

        return synth_out
    


def harmonic_synth(pitch, amplitudes, sampling_rate):

    omega = torch.cumsum(2 * torch.pi * pitch / sampling_rate, 1)

    n_harmonic = amplitudes.shape[-1]
    omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal

