'''
https://github.com/hyakuchiki/realtimeDDSP/blob/master/plot.py
'''

import os
from lightning import LightningModule, Trainer
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import Callback

import wandb

def plot_figure_spec_wave(x, name=None, step=None, sr=16000, plot_dir= '',save=False):
    """Plot spectrograms/waveforms of original/reconstructed audio

    Args:
        x (numpy array): [n_samples]
        x_tilde (numpy array): [n_samples]
        sr (int, optional): sample rate. Defaults to 16000.
        dir (str): plot directory.
        name (str, optional): file name.
        step (int, optional): no. of steps of training.
    """
    fig, axes = plt.subplots(1, 1, figsize=(64, 64), squeeze=False)
    axes[0, 0].specgram(x, Fs=sr, scale='dB')
    #axes[1, 0].plot(x)
    #axes[1, 0].set_ylim(-1,1)
    if save:
        fig.savefig(os.path.join(plot_dir,f'{step}_{name}.png'))
        plt.close(fig)
    return fig

def save_to_wandb(i, name, logger, orig_audio, resyn_audio, plot_num=4, sr=16000):
        orig_audio = orig_audio.squeeze().detach().cpu().numpy()
        resyn_audio = resyn_audio.squeeze().detach().cpu().numpy()
        plot_num = min(plot_num, orig_audio.shape[0])
        columns = ['Original (Image)', 'Synthesized (Image)', 'Original (Audio)', 'Synthesized (Audio)']
        data = []
        for x_i, y_i in list(zip(orig_audio[:plot_num], resyn_audio[:plot_num])):
            fig_origin = plot_figure_spec_wave(x_i, name="Original" ,sr=sr )
            fig_synth = plot_figure_spec_wave(y_i, name="Synthesized" ,sr=sr)
            data.append([wandb.Image(fig_origin), wandb.Image(fig_synth), wandb.Audio(x_i, sample_rate=sr), wandb.Audio(y_i, sample_rate=sr)])
        logger.log_table(name, columns, data)        

def save_to_wandb_kd(i, name, logger, orig_audio, resyn_audio, teacher_audio, plot_num=4, sr=16000):
        orig_audio = orig_audio.squeeze().detach().cpu().numpy()
        resyn_audio = resyn_audio.squeeze().detach().cpu().numpy()
        teacher_audio = teacher_audio.squeeze().detach().cpu().numpy()
        plot_num = min(plot_num, orig_audio.shape[0])
        columns = ['Original (Image)', 'Synthesized (Image)', 'Teacher (Image)', 'Original (Audio)', 'Synthesized (Audio)', 'Teacher (Audio)']
        data = []
        for x_i, y_i, t_i in list(zip(orig_audio[:plot_num], resyn_audio[:plot_num], teacher_audio[:plot_num])):
            fig_origin = plot_figure_spec_wave(x_i, name="Original" ,sr=sr )
            fig_synth = plot_figure_spec_wave(y_i, name="Synthesized" ,sr=sr)
            fig_teacher = plot_figure_spec_wave(t_i, name="Teacher" ,sr=sr)
            data.append([wandb.Image(fig_origin), wandb.Image(fig_synth), wandb.Image(fig_teacher), wandb.Audio(x_i, sample_rate=sr), wandb.Audio(y_i, sample_rate=sr), wandb.Audio(t_i, sample_rate=sr)])
        logger.log_table(name, columns, data)
# def save_to_board(i, name, writer, orig_audio, resyn_audio, plot_num=4, sr=16000):
#     orig_audio = orig_audio.unsqueeze(0).detach().cpu().numpy()
#     resyn_audio = resyn_audio.unsqueeze(0).detach().cpu().numpy()
#     plot_num = min(plot_num, orig_audio.shape[0])
#     for j in range(plot_num):
#         writer.log_audio('{0}_orig/{1}'.format(name, j), orig_audio[j], i, sample_rate=sr)
#         writer.log_audio('{0}_resyn/{1}'.format(name, j), resyn_audio[j], i, sample_rate=sr)
#     fig = plot_recons(orig_audio.detach().cpu().numpy(), resyn_audio.detach().cpu().numpy(), '', sr=sr, num=plot_num, save=False)
#     writer.log_image('plot_recon_{0}'.format(name), fig, i)



class AudioLogger(Callback):
    def __init__(self, batch_frequency=1000, sr=16000):
        super().__init__()
        self.step_freq = batch_frequency
        self.sr = sr

    @rank_zero_only
    def log_local(self, writer, name, current_epoch, orig_audio, resyn_audio, teacher_audio=None):
        if teacher_audio is not None:
            save_to_wandb_kd(current_epoch, name, writer, orig_audio, resyn_audio, teacher_audio, plot_num=4, sr=self.sr)
        else:    
            save_to_wandb(current_epoch, name, writer, orig_audio, resyn_audio, plot_num=4, sr=self.sr)
        
    def log_audio(self, pl_module, batch, batch_idx, name="train"):
        if batch_idx % self.step_freq == 0:
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            # get audio
            with torch.no_grad():
                resyn_audio, teacher_audio = pl_module(batch)
            resyn_audio = torch.clamp(resyn_audio['synth_audio'].detach().cpu(), -1, 1)
            orig_audio = torch.clamp(batch['audio'].detach().cpu(), -1, 1)
            if teacher_audio is not None:
                teacher_audio = torch.clamp(teacher_audio['synth_audio'].detach().cpu(), -1, 1)
                orig_audio = torch.clamp(batch['audio'].detach().cpu(), -1, 1)
                
            self.log_local(pl_module.logger, name, pl_module.current_epoch, orig_audio, resyn_audio, teacher_audio)

            if is_train:
                pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pass
        # self.log_audio(pl_module, batch, batch_idx, name="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        #self.log_audio(pl_module, batch, batch_idx, name="val_"+str(dataloader_idx))
        pass
    
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_audio(pl_module, batch, batch_idx, name="test")