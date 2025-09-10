import argparse
import os
from typing import Dict, Tuple

import numpy as np
import soundfile as sf
import torch
import librosa
import torchcrepe
from omegaconf import OmegaConf
import hydra
import pyrootutils


# Ensure project root is on PYTHONPATH for `src` imports and Hydra targets
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


DB_RANGE = 80.0
F0_MIDI_RANGE = 127.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer audio from checkpoint using DDSP model")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("--input_wav", type=str, required=True, help="Path to input wav file")
    parser.add_argument("--output_wav", type=str, required=True, help="Path to output wav file")
    parser.add_argument(
        "--model_config",
        type=str,
        default="/home/huaxiao/distilling-ddsp/configs/model/ddx7/full_ddx7_fmflute.yaml",
        help="Path to model config yaml (_target_ based)",
    )
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--frame_rate", type=int, default=250, help="Feature frame rate (frames per second)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def hz_to_midi_torch(frequencies: torch.Tensor) -> torch.Tensor:
    notes = 12.0 * (torch.log2(torch.clamp(frequencies, min=1e-7)) - torch.log2(torch.tensor(440.0, device=frequencies.device))) + 69.0
    notes = torch.where(frequencies <= 0.0, torch.zeros_like(frequencies), notes)
    return notes


def extract_features(
    y: np.ndarray,
    sample_rate: int,
    frame_rate: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute f0 (Hz) with torchcrepe and loudness (dB) with librosa RMS.

    Returns tensors shaped [T] for both features.
    """
    hop_length = sample_rate // frame_rate  # 16000/250 = 64

    # f0 with torchcrepe
    audio_t = torch.from_numpy(y).float().to(device)
    if audio_t.ndim == 1:
        audio_t = audio_t.unsqueeze(0)  # [1, T]
    f0_hz = torchcrepe.predict(
        audio_t,
        sample_rate,
        hop_length,
        fmin=50.0,
        fmax=2000.0,
        model="full",
        batch_size=2048,
        device=device,
        return_periodicity=False,
    ).squeeze(0)  # [T]

    # loudness via RMS -> dB in [-inf, 0], then clip to [-80, 0]
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length, center=True).squeeze(0)
    loudness_db = librosa.amplitude_to_db(rms, ref=1.0)
    loudness_db = np.clip(loudness_db, -DB_RANGE, 0.0)

    # align lengths (crepe and librosa can differ by 1 frame)
    T = min(f0_hz.shape[0], loudness_db.shape[0])
    f0_hz = f0_hz[:T]
    loudness_db = loudness_db[:T]

    return f0_hz.detach(), torch.from_numpy(loudness_db).float().to(device)


def scale_features(f0_hz: torch.Tensor, loudness_db: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # f0: Hz -> MIDI -> [0, 1]
    f0_midi = hz_to_midi_torch(f0_hz)
    f0_scaled = f0_midi / F0_MIDI_RANGE

    # loudness: [-80, 0] -> [0, 1]
    loudness_scaled = (loudness_db / DB_RANGE) + 1.0
    return f0_scaled, loudness_scaled


def build_model(model_config_path: str, device: torch.device) -> torch.nn.Module:
    cfg = OmegaConf.load(model_config_path)
    model: torch.nn.Module = hydra.utils.instantiate(cfg)
    model = model.to(device)
    model.eval()
    return model


def load_weights_into_model(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    """Load only the nested `model.*` weights from Lightning checkpoint into bare model."""
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    renamed: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            renamed[k[len("model."):]] = v
    missing, unexpected = model.load_state_dict(renamed, strict=False)
    if len(missing) > 0:
        print(f"[WARN] Missing keys: {len(missing)} (showing up to 5): {missing[:5]}")
    if len(unexpected) > 0:
        print(f"[WARN] Unexpected keys: {len(unexpected)} (showing up to 5): {unexpected[:5]}")


def run_inference(
    ckpt_path: str,
    input_wav: str,
    output_wav: str,
    model_config: str,
    sample_rate: int,
    frame_rate: int,
    device_str: str,
):
    device = torch.device(device_str)

    # Load audio
    y, sr = librosa.load(input_wav, sr=sample_rate, mono=True)
    y = librosa.util.normalize(y)

    # Build model and load weights
    model = build_model(model_config, device)
    load_weights_into_model(model, ckpt_path, device)

    # Features
    f0_hz, loudness_db = extract_features(y, sample_rate, frame_rate, device)
    f0_scaled, loudness_scaled = scale_features(f0_hz, loudness_db)

    # Pack inputs (B, T, 1)
    x = {
        "f0": f0_hz.unsqueeze(-1).unsqueeze(0),
        "f0_scaled": f0_scaled.unsqueeze(-1).unsqueeze(0),
        "loudness_scaled": loudness_scaled.unsqueeze(-1).unsqueeze(0),
    }

    # Synthesize
    with torch.no_grad():
        out: Dict[str, torch.Tensor] = model(x)
        synth_audio = out["synth_audio"].squeeze(0).detach().cpu().numpy()

    # Save
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)
    sf.write(output_wav, synth_audio, samplerate=sample_rate)
    print(f"Saved synthesized audio to: {output_wav}")


def main():
    args = parse_args()
    run_inference(
        ckpt_path=args.ckpt_path,
        input_wav=args.input_wav,
        output_wav=args.output_wav,
        model_config=args.model_config,
        sample_rate=args.sample_rate,
        frame_rate=args.frame_rate,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()


