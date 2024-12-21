import os, glob, pickle, itertools, logging
import hydra
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import torchaudio
import torch
import warnings
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

#from dataset.spectral_ops import calc_f0, FMIN, FMAX
#from dataset.spectral_ops import calc_loudness
import torchcrepe
from src.data.preprocessor import F0LoudnessRMSPreprocessor
from src.models.ddsp_decoder import DDSP_Decoder



DB_RANGE = 80.0
FMIN = 50
FMAX = 2000

# logging.basicConfig()
# log = logging.getLogger(__name__)
# log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

def pad_or_trim_to_expected_length(vector: torch.Tensor, expected_len: int, pad_value: float=0.0, len_tolerance: int=20):
    """Ported from DDSP
    Make vector equal to the expected length.

    Feature extraction functions like `compute_loudness()` or `compute_f0` produce feature vectors that vary in length depending on factors such as `sample_rate` or `hop_size`. This function corrects vectors to the expected length, warning the user if the difference between the vector and expected length was unusually high to begin with.

    Args:
        vector: Tensor. Shape [(batch,) vector_length]
        expected_len: Expected length of vector.
        pad_value: If float, value to pad at end of vector else pad_mode ('reflect', 'replicate')
        len_tolerance: Tolerance of difference between original and desired vector length.

    Returns:
        vector: Vector with corrected length.

    Raises:
        ValueError: if `len(vector)` is different from `expected_len` beyond
        `len_tolerance` to begin with.
    """
    expected_len = int(expected_len)
    vector_len = int(vector.shape[-1])

    if abs(vector_len - expected_len) > len_tolerance:
        # Ensure vector was close to expected length to begin with
        raise ValueError('Vector length: {} differs from expected length: {} '
                        'beyond tolerance of : {}'.format(vector_len,
                                                        expected_len,
                                                        len_tolerance))

    is_1d = (len(vector.shape) == 1)
    vector = vector[None, :] if is_1d else vector

    # Pad missing samples
    if vector_len < expected_len:
        n_padding = expected_len - vector_len
        if isinstance(pad_value, str):
            vector = F.pad(vector, ((0, 0, 0, n_padding)), mode=pad_value)
        else:
            vector = F.pad(vector, ((0, 0, 0, n_padding)), mode='constant', value=pad_value)
    # Trim samples
    elif vector_len > expected_len:
        vector = vector[..., :expected_len]

    # Remove temporary batch dimension.
    vector = vector[0] if is_1d else vector
    return vector

def spec_loudness(spec, a_weighting: torch.Tensor, range_db:float=DB_RANGE, ref_db:float=0.0):
    """
    Args:
        spec: Shape [..., freq_bins]
    """
    power = spec.real**2+spec.imag**2
    weighting = 10**(a_weighting/10) #db to linear
    weighted_power = power * weighting
    avg_power = torch.mean(weighted_power, dim=-1)
    # to db
    min_power = 10**-(range_db / 10.0)
    power = torch.clamp(avg_power, min=min_power)
    db = 10.0 * torch.log10(power)
    db -= ref_db
    db = torch.clamp(db, min=-range_db)
    return db

def A_weighting(frequencies, min_db=-80.0):
    # ported from librosa
    f_sq = np.asanyarray(frequencies) ** 2.0
    const = np.array([12194.217, 20.598997, 107.65265, 737.86223]) ** 2.0
    weights = 2.0 + 20.0 * (
        np.log10(const[0])
        + 2 * np.log10(f_sq)
        - np.log10(f_sq + const[0])
        - np.log10(f_sq + const[1])
        - 0.5 * np.log10(f_sq + const[2])
        - 0.5 * np.log10(f_sq + const[3])
    )
    return weights if min_db is None else np.maximum(min_db, weights)

def fft_frequencies(*, sr=22050, n_fft=2048):
    # ported from librosa
    return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

def compute_loudness(audio, sample_rate=16000, frame_rate=50, n_fft=2048, range_db=DB_RANGE, ref_db=0.0, a_weighting=None, center=True):
    """Perceptual loudness in dB, relative to white noise, amplitude=1.

    Args:
        audio: tensor. Shape [batch_size, audio_length] or [audio_length].
        sample_rate: Audio sample rate in Hz.
        frame_rate: Rate of loudness frames in Hz.
        n_fft: Fft window size.
        range_db: Sets the dynamic range of loudness in decibels. The minimum loudness (per a frequency bin) corresponds to -range_db.
        ref_db: Sets the reference maximum perceptual loudness as given by (A_weighting + 10 * log10(abs(stft(audio))**2.0).

    Returns:
        Loudness in decibels. Shape [batch_size, n_frames] or [n_frames,].
    """
    # Temporarily a batch dimension for single examples.
    is_1d = (len(audio.shape) == 1)
    if is_1d:
        audio = audio[None, :]

    # Take STFT.
    hop_length = sample_rate // frame_rate
    s = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, return_complex=True, center=center)
    # batch, frequency_bins, n_frames
    s = s.permute(0, 2, 1)
    if a_weighting is None:
        frequencies = fft_frequencies(sr=sample_rate, n_fft=n_fft)
        a_weighting = A_weighting(frequencies+1e-8)
        a_weighting = torch.from_numpy(a_weighting.astype(np.float32)).to(audio.device)
    loudness = spec_loudness(s, a_weighting, range_db, ref_db)

    # Remove temporary batch dimension.
    loudness = loudness[0] if is_1d else loudness

    # Compute expected length of loudness vector
    n_secs = audio.shape[-1] / float(sample_rate)  # `n_secs` can have milliseconds
    expected_len = int(n_secs * frame_rate)

    # Pad with `-range_db` noise floor or trim vector
    #loudness = pad_or_trim_to_expected_length(loudness, expected_len, -range_db)
    return loudness

def compute_f0(audio, sample_rate, frame_rate, center=True, f0_range=(FMIN, FMAX), viterbi=True):
    """ For preprocessing
    Args:
        audio: torch.Tensor of single audio example. Shape [audio_length,].
        sample_rate: Sample rate in Hz.

    Returns:
        f0_hz: Fundamental frequency in Hz. Shape [n_frames]
        periodicity: Basically, confidence of pitch value. Shape [n_frames]
    """
    audio = audio[None, :]

    hop_length = sample_rate // frame_rate
    # Compute f0 with torchcrepe.
    # uses viterbi by default
    # pad=False is probably center=False
    # [output_shape=(1, 1 + int(time // hop_length))]
    f0_dec = torchcrepe.decode.viterbi if viterbi else torchcrepe.decode.argmax
    with torch.no_grad():
        try:
            f0_hz, periodicity = torchcrepe.predict(audio, sample_rate, hop_length=hop_length, pad=center, device='cuda', batch_size=64, model='full', fmin=f0_range[0], fmax=f0_range[1], return_periodicity=True, decoder=f0_dec)
        except:
            warnings.warn("GPU not available, detecting f0 on cpu (SLOW)", ResourceWarning)
            f0_hz, periodicity = torchcrepe.predict(audio, sample_rate, hop_length=hop_length, pad=center, device='cpu', batch_size=64, model='full', fmin=f0_range[0], fmax=f0_range[1], return_periodicity=True, decoder=f0_dec)

    f0_hz = f0_hz[0]
    periodicity = periodicity[0]

    n_secs = audio.shape[-1] / float(sample_rate)  # `n_secs` can have milliseconds
    expected_len = int(n_secs * frame_rate)
    #f0_hz = pad_or_trim_to_expected_length(f0_hz, expected_len, 'replicate')
    return f0_hz, periodicity

class SliceDataset(Dataset):
    # slice length [s] sections from longer audio files like urmp 
    # some of LMDB code borrowed from UDLS 
    # https://github.com/caillonantoine/UDLS/tree/7a99c503eb02ca60852626ca0542ddc1117295ac (A. Caillon, MIT License)
    # and https://github.com/rmccorm4/PyTorch-LMDB/blob/master/folder2lmdb.py
    def __init__(self, raw_dir, output_dir, sample_rate=16000, length=4.0, frame_rate=250, f0_range=(FMIN, FMAX), f0_viterbi=True):
        self.raw_dir = raw_dir
        self.sample_rate = sample_rate
        self.length = length
        self.hop_size = 64
        self.frame_rate = frame_rate
        self.f0_range = f0_range
        self.f0_viterbi = f0_viterbi
        self.device = 'cpu'
        self.center = False
        assert sample_rate % frame_rate == 0, 'sample_rate must be divisible by frame_rate'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        self.output_dir = output_dir
        self.preprocess = F0LoudnessRMSPreprocessor()

    def calculate_features(self, audio):
        # calculate f0 and loudness
        # pad=True->center=True
        x ={}
        pitch, periodicity = compute_f0(audio, self.sample_rate, frame_rate=self.frame_rate, center=False, f0_range=self.f0_range, viterbi=self.f0_viterbi)
        loudness = compute_loudness(audio, self.sample_rate, frame_rate=self.frame_rate, n_fft=2048, center=False)
        #pitch, confidence = calc_f0(audio, rate=self.sample_rate, hop_size=self.hop_size, fmin=FMIN, fmax=FMAX, model='full', batch_size=128, device=self.device, center=self.center)
        #loudness = calc_loudness(audio, rate=self.sample_rate, n_fft=2048, hop_size=self.hop_size, center=self.center)
        x['f0'] = (torch.from_numpy(pitch)).unsqueeze(1)
        x['loudness'] = (torch.from_numpy(loudness)).unsqueeze(1)
        return x

    def process(self, model):
        self.raw_files = sorted(list(itertools.chain(*(glob.glob(os.path.join(self.raw_dir, f'**/*.{ext}'), recursive=True) for ext in ['mp3', 'wav', 'MP3', 'WAV']))))
        # load audio
        resample = {}
        for audio_file in tqdm(self.raw_files):
            print("Processing: ", audio_file)
            try:
                audio, orig_sr = torchaudio.load(audio_file)
                audio = audio.mean(dim=0) # force mono
            except RuntimeError:
                warnings.warn('Falling back to librosa because torchaudio loading (sox) failed.')
                import librosa
                audio, orig_sr = librosa.load(audio_file, sr=None, mono=True)
                audio = torch.from_numpy(audio)
            # resample
            if orig_sr != self.sample_rate:
                if orig_sr not in resample:
                    # save kernel
                    resample[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.sample_rate, resampling_method='kaiser_window', lowpass_filter_width=64, rolloff=0.99)
                audio = resample[orig_sr](audio)
            # pad so that it can be evenly sliced
            len_audio_chunk = int(self.sample_rate*self.length)
            pad_audio = (len_audio_chunk - (audio.shape[-1] % len_audio_chunk)) % len_audio_chunk
            audio = F.pad(audio, (0, pad_audio))
            # split 
            audios = torch.split(audio, len_audio_chunk)
            
            outputs = []
            for x in tqdm(audios):
                # calculate f0 and loudness
                x = self.calculate_features(x)
                # unsqueeze to add batch dimension
                x = {k: v.unsqueeze(0) for k, v in x.items()}
                # preprocess
                x = self.preprocess.run(x)
                # model forward
                y = model(x)
                outputs.append(y['synth_audio'])
                
            # save
            resynth_audio = ((torch.cat(outputs, dim=-1)).flatten()).unsqueeze(0)
            output_file = os.path.join(self.output_dir, os.path.basename(audio_file))
            print("Saving: ", output_file)
            torchaudio.save(output_file, resynth_audio, self.sample_rate)
                
      

def remove_keys(mydict):
         return dict((k.removeprefix('model.'), remove_keys(v) if hasattr(v,'keys') else v) for k,v in mydict.items())

def load_model(model_cfg, model_path) -> DDSP_Decoder:
    """Loads a pre-trained model from a checkpoint.

    Args:
        model_path (str): Path to the checkpoint.

    Returns:
        DDSP_Decoder: The model.
    """
    model = hydra.utils.instantiate(model_cfg)
    model_dict = torch.load(model_path)
    model.load_state_dict(remove_keys(model_dict['state_dict']), strict=False)
    model.eval()
    return model


def main():
    model_cfg = OmegaConf.load('/home/greg/PythonProjects/Compression_Algorithms_DDSP/DDSP/MyProjects/Compression_Algorithm_DDSP/configs/model/ddsp/large_hpn.yaml')
    model_path = "/home/greg/PythonProjects/Compression_Algorithms_DDSP/DDSP/MyProjects/Compression_Algorithm_DDSP/logs/large/runs/2024-02-27_17-36-19/checkpoints/epoch_13891.ckpt"
    model = load_model(model_cfg, model_path)
    
    
    dataset = SliceDataset(raw_dir='/home/greg/PythonProjects/Compression_Algorithms_DDSP/DDSP/MyProjects/Compression_Algorithm_DDSP/dataset/files/train16/flute_test', 
                           output_dir='/home/greg/PythonProjects/Compression_Algorithms_DDSP/DDSP/MyProjects/Compression_Algorithm_DDSP/dataset/files/train16/flute_resynth')
    
    # process
    dataset.process(model)
    
if __name__ == '__main__':
    main()
                
                