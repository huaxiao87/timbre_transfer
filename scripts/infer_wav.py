import argparse
import os
from typing import Dict, Tuple

import numpy as np
import soundfile as sf
import torch
import librosa
from omegaconf import OmegaConf
import hydra
import pyrootutils

try:
    import torchcrepe
except Exception:
    torchcrepe = None

try:
    from ddx7.spectral_ops import calc_loudness as ddx7_calc_loudness, calc_f0 as ddx7_calc_f0
except Exception:
    ddx7_calc_loudness = None
    ddx7_calc_f0 = None


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
    # Pitch extraction options
    parser.add_argument("--pitch_method", type=str, choices=["ddx7", "torchcrepe"], default="ddx7",
                        help="Use ddx7.spectral_ops (recommended) or torchcrepe")
    parser.add_argument("--fmin", type=float, default=50.0)
    parser.add_argument("--fmax", type=float, default=1200.0)
    parser.add_argument("--crepe_model", type=str, default="full", choices=["full", "tiny"])
    parser.add_argument("--periodicity_threshold", type=float, default=0.2,
                        help="For torchcrepe: zero-out f0 where periodicity < threshold")
    return parser.parse_args()


def hz_to_midi_torch(frequencies: torch.Tensor) -> torch.Tensor:
    notes = 12.0 * (torch.log2(torch.clamp(frequencies, min=1e-7)) - torch.log2(torch.tensor(440.0, device=frequencies.device))) + 69.0
    notes = torch.where(frequencies <= 0.0, torch.zeros_like(frequencies), notes)
    return notes


def extract_features_ddx7(
    y: np.ndarray,
    sample_rate: int,
    frame_rate: int,
    fmin: float,
    fmax: float,
    crepe_model: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert ddx7_calc_f0 is not None and ddx7_calc_loudness is not None, "ddx7.spectral_ops is not available"
    hop_length = sample_rate // frame_rate
    # ddx7 f0 returns (f0, confidence)
    f0_np, conf_np = ddx7_calc_f0(y, rate=sample_rate, hop_size=hop_length,
                                  fmin=fmin, fmax=fmax, model=crepe_model,
                                  batch_size=2048, device=device, center=True)
    # Zero-out low confidence frames (match training behavior more closely)
    conf_thr = 0.2
    f0_np = np.where(conf_np < conf_thr, 0.0, f0_np)

    # Loudness
    loudness_np = ddx7_calc_loudness(y, rate=sample_rate, n_fft=2048, hop_size=hop_length, center=True)
    loudness_np = np.clip(loudness_np, -DB_RANGE, 0.0)

    T = min(f0_np.shape[0], loudness_np.shape[0])
    f0_np = f0_np[:T]
    loudness_np = loudness_np[:T]
    return torch.from_numpy(f0_np).float().to(device), torch.from_numpy(loudness_np).float().to(device)


def extract_features_torchcrepe(
    y: np.ndarray,
    sample_rate: int,
    frame_rate: int,
    fmin: float,
    fmax: float,
    crepe_model: str,
    periodicity_threshold: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert torchcrepe is not None, "torchcrepe is not installed"
    hop_length = sample_rate // frame_rate

    audio_t = torch.from_numpy(y).float().to(device)
    if audio_t.ndim == 1:
        audio_t = audio_t.unsqueeze(0)  # [1, T]
    # Viterbi decoding improves continuity; request periodicity
    f0_hz, periodicity = torchcrepe.predict(
        audio_t,
        sample_rate,
        hop_length,
        fmin=fmin,
        fmax=fmax,
        model=crepe_model,
        batch_size=2048,
        device=device,
        decoder=torchcrepe.decode.viterbi,
        return_periodicity=True,
    )
    f0_hz = f0_hz.squeeze(0)  # [T]
    periodicity = periodicity.squeeze(0)
    # Simple smoothing and thresholding
    if hasattr(torchcrepe, "filter"):
        try:
            f0_hz = torchcrepe.filter.median(f0_hz, 3)
            periodicity = torchcrepe.filter.mean(periodicity, 3)
        except Exception:
            pass
    f0_hz = torch.where(periodicity >= periodicity_threshold, f0_hz, torch.zeros_like(f0_hz))

    # loudness via RMS -> dB in [-inf, 0], then clip to [-80, 0]
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length, center=True).squeeze(0)
    loudness_db = librosa.amplitude_to_db(rms, ref=1.0)
    loudness_db = np.clip(loudness_db, -DB_RANGE, 0.0)

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
    pitch_method: str,
    fmin: float,
    fmax: float,
    crepe_model: str,
    periodicity_threshold: float,
):
    device = torch.device(device_str)

    # Load audio
    y, sr = librosa.load(input_wav, sr=sample_rate, mono=True)
    y = librosa.util.normalize(y)

    # Build model and load weights
    model = build_model(model_config, device)
    load_weights_into_model(model, ckpt_path, device)

    # Features
    use_ddx7 = (pitch_method == "ddx7") and (ddx7_calc_f0 is not None and ddx7_calc_loudness is not None)
    use_crepe = (pitch_method == "torchcrepe") and (torchcrepe is not None)

    if not use_ddx7 and not use_crepe:
        # fallback: prefer ddx7 if available else torchcrepe
        use_ddx7 = ddx7_calc_f0 is not None and ddx7_calc_loudness is not None
        use_crepe = not use_ddx7 and (torchcrepe is not None)

    if use_ddx7:
        f0_hz, loudness_db = extract_features_ddx7(
            y=y,
            sample_rate=sample_rate,
            frame_rate=frame_rate,
            fmin=fmin,
            fmax=fmax,
            crepe_model=crepe_model,
            device=device,
        )
    elif use_crepe:
        f0_hz, loudness_db = extract_features_torchcrepe(
            y=y,
            sample_rate=sample_rate,
            frame_rate=frame_rate,
            fmin=fmin,
            fmax=fmax,
            crepe_model=crepe_model,
            periodicity_threshold=periodicity_threshold,
            device=device,
        )
    else:
        raise RuntimeError("Neither ddx7.spectral_ops nor torchcrepe are available for pitch extraction")
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
        pitch_method=args.pitch_method,
        fmin=args.fmin,
        fmax=args.fmax,
        crepe_model=args.crepe_model,
        periodicity_threshold=args.periodicity_threshold,
    )


if __name__ == "__main__":
    main()


