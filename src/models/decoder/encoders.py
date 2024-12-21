"""
Z-encoder
The code mainly comes from https://github.com/ben-hayes/ddsp-autoencoder-torch with minor adaptations.
"""

import torch
import torch.nn as nn
from torchaudio.transforms import MFCC

SAMPLE_RATE = 16000
SAMPLE_LENGTH_IN_SECONDS = 4

# Create a class Z-Encoder that will be used to encode the audio
class Zencoder(nn.Module):
    """
    The recurrent encoder module.
    """
    def __init__(
            self,
            in_size_in_seconds=SAMPLE_LENGTH_IN_SECONDS,
            sr=SAMPLE_RATE,
            n_mfcc=30,
            n_fft=1024,
            hop_length=64,
            n_mels=128,
            hidden_size=512,
            z_size=16):
        """
        Construct an instance of Zencoder

        Args:
            in_size_in_seconds (float, optional): The length of the input in
                seconds. Defaults to SAMPLE_LENGTH_IN_SECONDS.
            sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
            n_mfcc (int, optional): Number of MFCCs. Defaults to 30.
            n_fft (int, optional): FFT size. Defaults to 1024.
            hop_length (int, optional): FFT hop length. Defaults to 256.
            n_mels (int, optional): Number of mel bands. Defaults to 128.
            rnn_dim (int, optional): Number of RNN states. Defaults to 512.
            z_size (int, optional): Size of latent dimension. Defaults to 16.
        """
        super().__init__()
        self.sr = sr
        self.in_size = sr * in_size_in_seconds
        self.mfcc = MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": n_mels,
                "hop_length": hop_length,
                "f_min": 20.0,
                "f_max": 8000.0
            })

        self.seq_len = int(self.in_size // hop_length) 
        self.norm = nn.LayerNorm(n_mfcc)
        self.gru = nn.GRU(input_size=n_mfcc, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, z_size)

    def forward(self, x):
        """
        Encode an audio sample to a latent vector

        Args:
            x (torch.Tensor): Batch of audio samples

        Returns:
            torch.Tensor: Time distributed latent code
        """
        # Check if the input is batch,time,channels and if not. Remove channels
        if len(x.shape) > 2:
            x = x.squeeze(2)
        z = self.mfcc(x)
        # if self.training:
            # add newaxis to z to make it 4D
        z = z.unsqueeze(2)
            # Interpolate mfcc frame dimension to 500
        z = nn.functional.interpolate(z, size=(1,self.seq_len), mode='bilinear')
            # Remove newaxis
        z = z.squeeze(2)
            
        z = self.norm(z.permute(0,2,1))
        # z = z.permute(2, 0, 1)
        z, _ = self.gru(z)
        z = self.linear(z)
        return z