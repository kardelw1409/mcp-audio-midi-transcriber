from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("mcp_audio_midi_transcriber")

mcp = FastMCP("audio_midi_transcriber")


StemType = Literal["drums", "bass", "vocals", "other"]


@dataclass
class GridConfig:
    tempo_bpm: float
    grid_start_seconds: float
    start_bar: int
    end_bar: int | None
    steps_per_bar: int


@dataclass
class HardwareInfo:
    torch_cuda_available: bool
    torch_cuda_version: str | None
    torch_device_name: str | None
    tensorflow_gpus: list[str]
    tensorflow_error: str | None


@dataclass
class TranscriptionResult:
    stem_type: str
    output_dir: Path
    midi_files: dict[str, str]
    manifest_path: str
    notes: list[str]


###############################################################################
# Utility helpers
###############################################################################


def _configure_logging() -> None:
    if logger.handlers:
        return
    logging.basicConfig(
        level=os.environ.get("AUDIO_MIDI_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _validate_audio_path(path: Path) -> Path:
    abs_path = path.expanduser().resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"audio_path not found: {abs_path}")
    if not abs_path.is_file():
        raise ValueError(f"audio_path is not a file: {abs_path}")
    return abs_path


def _detect_hardware() -> HardwareInfo:
    torch_cuda_available = False
    torch_cuda_version: str | None = None
    torch_device_name: str | None = None
    tensorflow_gpus: list[str] = []
    tensorflow_error: str | None = None

    try:
        import torch

        torch_cuda_available = torch.cuda.is_available()
        torch_cuda_version = getattr(torch.version, "cuda", None)
        if torch_cuda_available:
            torch_device_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        logger.warning("Failed to inspect torch CUDA availability: %s", exc)

    try:
        import tensorflow as tf  # type: ignore

        tensorflow_gpus = [device.name for device in tf.config.list_physical_devices("GPU")]
    except Exception as exc:
        tensorflow_error = str(exc)

    return HardwareInfo(
        torch_cuda_available=torch_cuda_available,
        torch_cuda_version=torch_cuda_version,
        torch_device_name=torch_device_name,
        tensorflow_gpus=tensorflow_gpus,
        tensorflow_error=tensorflow_error,
    )


def _grid_step_seconds(config: GridConfig) -> float:
    beat_len = 60.0 / config.tempo_bpm
    bar_len = 4.0 * beat_len
    step = bar_len / float(config.steps_per_bar)
    if step <= 0:
        raise ValueError("steps_per_bar must be > 0")
    return step


def _grid_bounds(config: GridConfig) -> tuple[float, float | None]:
    if config.start_bar < 1:
        raise ValueError("start_bar must be >= 1")
    beat_len = 60.0 / config.tempo_bpm
    bar_len = 4.0 * beat_len
    start_time = config.grid_start_seconds + (config.start_bar - 1) * bar_len
    end_time = None
    if config.end_bar is not None:
        if config.end_bar < config.start_bar:
            raise ValueError("end_bar must be >= start_bar")
        end_time = config.grid_start_seconds + config.end_bar * bar_len
    return start_time, end_time


def _quantize_time(time_value: float, config: GridConfig) -> float:
    step = _grid_step_seconds(config)
    return round(time_value / step) * step


def _quantize_time_relative(time_value: float, config: GridConfig) -> float:
    step = _grid_step_seconds(config)
    relative = time_value - config.grid_start_seconds
    snapped = round(relative / step) * step
    return config.grid_start_seconds + snapped


def _quantize_duration(duration: float, config: GridConfig, min_seconds: float) -> float:
    step = _grid_step_seconds(config)
    snapped = max(min_seconds, round(duration / step) * step)
    return max(snapped, min_seconds)


def _clip_to_window(
    start: float,
    end: float,
    window_start: float | None,
    window_end: float | None,
) -> tuple[float, float] | None:
    if window_start is not None and end < window_start:
        return None
    if window_end is not None and start > window_end:
        return None
    start_clipped = max(start, window_start) if window_start is not None else start
    end_clipped = min(end, window_end) if window_end is not None else end
    if end_clipped <= start_clipped:
        return None
    return start_clipped, end_clipped


def _normalize_velocity(velocity: float, *, min_velocity: int) -> int:
    return int(min(127, max(min_velocity, round(velocity))))


def _sanitize_audio(y: Any) -> Any:
    import numpy as np

    if y is None:
        return y
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


###############################################################################
# Drum transcription
###############################################################################


def _bandpass_filter(y: Any, sr: int, low: float, high: float) -> Any:
    from scipy.signal import butter, sosfiltfilt

    if y is None or len(y) < 8:
        return y
    nyquist = 0.5 * sr
    low_norm = max(1e-5, low / nyquist)
    high_norm = min(0.999, high / nyquist)
    sos = butter(4, [low_norm, high_norm], btype="band", output="sos")
    return sosfiltfilt(sos, y)


def _peak_pick_times(
    flux: Any,
    sr: int,
    hop_length: int,
    *,
    threshold: float,
    min_separation_frames: int,
) -> list[float]:
    import numpy as np

    if flux is None or len(flux) == 0:
        return []
    flux = np.asarray(flux)
    flux = flux / (np.max(flux) + 1e-9)
    peaks: list[int] = []
    last_peak = -min_separation_frames
    for idx in range(1, len(flux) - 1):
        if flux[idx] < threshold:
            continue
        if flux[idx] >= flux[idx - 1] and flux[idx] >= flux[idx + 1]:
            if idx - last_peak >= min_separation_frames:
                peaks.append(idx)
                last_peak = idx
    return [(frame * hop_length) / sr for frame in peaks]


def _band_energy_flux(
    magnitude: Any,
    *,
    band_bins: slice,
) -> Any:
    import numpy as np

    band_energy = np.sum(magnitude[band_bins, :], axis=0)
    flux = np.maximum(0.0, np.diff(band_energy, prepend=band_energy[0]))
    return flux


def _transcribe_drums(
    *,
    audio_path: Path,
    config: GridConfig,
    onset_threshold: float,
    min_velocity: int,
    analysis_sr: int = 22050,
    drums_fast_mode: bool = True,
    snare_threshold: float = 0.14,
    snare_kick_ratio: float = 1.25,
) -> tuple[dict[str, list[tuple[float, float, int]]], list[str]]:
    import numpy as np
    import librosa

    window_start, window_end = _grid_bounds(config)
    offset = 0.0
    y, sr = librosa.load(path=str(audio_path), mono=True, sr=analysis_sr)
    y = _sanitize_audio(y)

    hop_length = 1024 if drums_fast_mode else 512
    n_fft = 2048 if drums_fast_mode else 1024

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    kick_bins = np.where((freqs >= 20.0) & (freqs <= 120.0))[0]
    snare_bins = np.where((freqs >= 180.0) & (freqs <= 2000.0))[0]
    hat_bins = np.where((freqs >= 2500.0) & (freqs <= 12000.0))[0]

    if len(kick_bins) == 0 or len(snare_bins) == 0 or len(hat_bins) == 0:
        return {"kick": [], "snare": [], "hat": []}, ["Drum band bins are empty; check SR/FFT settings."]

    kick_flux = _band_energy_flux(magnitude, band_bins=slice(kick_bins[0], kick_bins[-1] + 1))
    snare_flux = _band_energy_flux(magnitude, band_bins=slice(snare_bins[0], snare_bins[-1] + 1))
    hat_flux = _band_energy_flux(magnitude, band_bins=slice(hat_bins[0], hat_bins[-1] + 1))

    min_sep = 3 if drums_fast_mode else 2
    kick_onsets = _peak_pick_times(kick_flux, sr, hop_length, threshold=onset_threshold, min_separation_frames=min_sep)
    snare_onsets = _peak_pick_times(snare_flux, sr, hop_length, threshold=snare_threshold, min_separation_frames=min_sep)
    hat_onsets = _peak_pick_times(hat_flux, sr, hop_length, threshold=onset_threshold * 0.8, min_separation_frames=1)

    notes: dict[str, list[tuple[float, float, int]]] = {"kick": [], "snare": [], "hat": []}
    warnings: list[str] = []

    def _make_notes(onsets: list[float], label: str, base_velocity: int) -> None:
        for onset_time in onsets:
            start = max(0.0, float(onset_time) + float(offset))
            end = start + 0.08
            clipped = _clip_to_window(start, end, window_start, window_end)
            if clipped is None:
                continue
            start, end = clipped
            start = _quantize_time_relative(start, config)
            duration = _quantize_duration(end - start, config, min_seconds=0.05)
            end = start + duration
            if window_end is not None and end > window_end:
                end = window_end
            notes[label].append((start, end, base_velocity))

    _make_notes(kick_onsets, "kick", _normalize_velocity(110, min_velocity=min_velocity))

    # Snare gating: ignore snare onsets dominated by kick energy.
    snare_notes: list[tuple[float, float, int]] = []
    for onset_time in snare_onsets:
        frame = int(round(onset_time * sr / hop_length))
        frame = max(0, min(frame, len(kick_flux) - 1))
        if kick_flux[frame] * snare_kick_ratio >= snare_flux[frame]:
            continue
        snare_notes.append((float(onset_time) + float(offset), float(onset_time) + float(offset) + 0.08, _normalize_velocity(100, min_velocity=min_velocity)))

    for start, end, velocity in snare_notes:
        clipped = _clip_to_window(start, end, window_start, window_end)
        if clipped is None:
            continue
        start, end = clipped
        start = _quantize_time_relative(start, config)
        duration = _quantize_duration(end - start, config, min_seconds=0.05)
        end = start + duration
        if window_end is not None and end > window_end:
            end = window_end
        notes["snare"].append((start, end, velocity))
    _make_notes(hat_onsets, "hat", _normalize_velocity(80, min_velocity=min_velocity))

    if sum(len(v) for v in notes.values()) == 0:
        warnings.append("No drum onsets detected after STFT energy-flux detection.")

    return notes, warnings


###############################################################################
# Pitch transcription (bass/vocals)
###############################################################################


def _preprocess_bass(audio_path: Path) -> Path:
    import numpy as np
    import soundfile as sf
    import librosa

    y, sr = librosa.load(str(audio_path), mono=True)
    y = _sanitize_audio(y)
    y = _sanitize_audio(_bandpass_filter(y, sr, 30.0, 300.0))
    y = librosa.effects.preemphasis(y)
    y = _sanitize_audio(y)
    out_path = audio_path.with_suffix(".bass_temp.wav")
    sf.write(str(out_path), y.astype(np.float32), sr)
    return out_path


def _predict_basic_pitch(
    *,
    audio_path: Path,
    min_freq: float,
    max_freq: float,
    onset_threshold: float,
    frame_threshold: float,
    min_note_length: float,
) -> tuple[Any, list[str]]:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    notes: list[str] = []

    model_output, midi_data, _ = predict(
        str(audio_path),
        ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_length,
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
    )
    _ = model_output

    if midi_data is None:
        notes.append("Basic Pitch returned no MIDI data.")
    return midi_data, notes


def _clean_pitch_notes(
    midi_data: Any,
    *,
    window_start: float | None,
    window_end: float | None,
    config: GridConfig,
    min_velocity: int,
    min_note_length: float,
    monophonic: bool,
) -> list[Any]:
    if midi_data is None:
        return []

    notes = []
    for instrument in getattr(midi_data, "instruments", []):
        for note in instrument.notes:
            if note.velocity < min_velocity:
                continue
            duration = note.end - note.start
            if duration < min_note_length:
                continue

            clipped = _clip_to_window(note.start, note.end, window_start, window_end)
            if clipped is None:
                continue
            start, end = clipped
            start = _quantize_time(start, config)
            duration = _quantize_duration(end - start, config, min_seconds=min_note_length)
            end = start + duration
            note.start = start
            note.end = end
            note.velocity = _normalize_velocity(note.velocity, min_velocity=min_velocity)
            notes.append(note)

    notes.sort(key=lambda n: n.start)

    if monophonic:
        mono_notes = []
        last_end = -1.0
        for note in notes:
            if note.start < last_end and mono_notes:
                mono_notes[-1].end = max(mono_notes[-1].start + min_note_length, note.start)
            if note.end <= note.start:
                continue
            mono_notes.append(note)
            last_end = note.end
        return mono_notes

    return notes


###############################################################################
# MIDI writing
###############################################################################


def _write_midi(
    *,
    output_path: Path,
    notes: list[tuple[float, float, int]],
    pitch: int,
    is_drum: bool,
    program: int | None = None,
    config: GridConfig,
) -> None:
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(initial_tempo=config.tempo_bpm)
    instrument = pretty_midi.Instrument(program=program or 0, is_drum=is_drum)
    for start, end, velocity in notes:
        instrument.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
    midi.instruments.append(instrument)
    midi.write(str(output_path))


def _write_midi_notes(
    *,
    output_path: Path,
    midi_notes: list[Any],
    program: int,
    is_drum: bool = False,
    config: GridConfig,
) -> None:
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(initial_tempo=config.tempo_bpm)
    instrument = pretty_midi.Instrument(program=program, is_drum=is_drum)
    for note in midi_notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(note.velocity),
                pitch=int(note.pitch),
                start=float(note.start),
                end=float(note.end),
            )
        )
    midi.instruments.append(instrument)
    midi.write(str(output_path))


###############################################################################
# Manifest writing
###############################################################################


def _write_manifest(
    *,
    output_dir: Path,
    hardware: HardwareInfo,
    model_info: dict[str, Any],
    midi_files: dict[str, str],
    notes: list[str],
) -> str:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "torch_cuda_available": hardware.torch_cuda_available,
            "torch_cuda_version": hardware.torch_cuda_version,
            "torch_device_name": hardware.torch_device_name,
            "tensorflow_gpus": hardware.tensorflow_gpus,
            "tensorflow_error": hardware.tensorflow_error,
        },
        "models": model_info,
        "midi_files": midi_files,
        "notes": notes,
    }

    manifest_path = output_dir / "transcription_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return str(manifest_path)


###############################################################################
# MCP tool
###############################################################################


@mcp.tool()
def transcribe_audio(
    audio_path: str,
    stem_type: StemType,
    output_dir: str = "midi_transcribed",
    tempo: float | None = None,
    grid_start_seconds: float | None = None,
    start_bar: int = 1,
    end_bar: int | None = None,
    bars: dict[str, int] | None = None,
    steps_per_bar: int = 16,
    section_name: str | None = None,
    onset_threshold: float = 0.1,
    frame_threshold: float = 0.3,
    min_note_length: float = 0.05,
    min_velocity: int = 28,
    drums_fast_mode: bool = True,
) -> dict[str, Any]:
    """
    Transcribe an audio file into MIDI, tailored for a specific stem type.
    Returns MIDI file paths plus a transcription_manifest.json with metadata.
    """
    _configure_logging()
    notes: list[str] = []

    try:
        if tempo is None or tempo <= 0:
            raise ValueError("tempo is required and must be > 0")
        if grid_start_seconds is None:
            raise ValueError("grid_start_seconds is required for grid alignment")
        if steps_per_bar <= 0:
            raise ValueError("steps_per_bar must be > 0")
        if min_note_length <= 0:
            raise ValueError("min_note_length must be > 0")
        if not 0 <= min_velocity <= 127:
            raise ValueError("min_velocity must be in range 0..127")

        input_path = _validate_audio_path(Path(audio_path))
        output_dir_path = Path(output_dir).expanduser().resolve()
        _ensure_dir(output_dir_path)

        if bars is not None:
            start_bar = int(bars.get("start_bar", start_bar))
            end_bar = bars.get("end_bar", end_bar)
            end_bar = int(end_bar) if end_bar is not None else None

        section_label = section_name or (
            f"bars_{start_bar}_{end_bar}" if end_bar is not None else f"bars_{start_bar}_end"
        )

        config = GridConfig(
            tempo_bpm=tempo,
            grid_start_seconds=grid_start_seconds,
            start_bar=start_bar,
            end_bar=end_bar,
            steps_per_bar=steps_per_bar,
        )

        hardware = _detect_hardware()

        if not hardware.torch_cuda_available:
            logger.warning("CUDA not available in torch; GPU acceleration may be unavailable.")
        if hardware.tensorflow_gpus:
            logger.info("TensorFlow GPU devices detected: %s", hardware.tensorflow_gpus)
        elif hardware.tensorflow_error:
            logger.warning("TensorFlow GPU check failed: %s", hardware.tensorflow_error)
        else:
            logger.warning("No TensorFlow GPU devices detected; Basic Pitch will run on CPU.")

        midi_files: dict[str, str] = {}

        if stem_type == "drums":
            drum_dir = output_dir_path / "drums"
            _ensure_dir(drum_dir)
            drum_notes, warnings = _transcribe_drums(
                audio_path=input_path,
                config=config,
                onset_threshold=onset_threshold,
                min_velocity=min_velocity,
                drums_fast_mode=drums_fast_mode,
            )
            notes.extend(warnings)

            mapping = {
                "kick": 36,
                "snare": 38,
                "hat": 42,
            }
            for label, pitch in mapping.items():
                if not drum_notes[label]:
                    continue
                out_path = drum_dir / f"{section_label}_{label}.mid"
                _write_midi(output_path=out_path, notes=drum_notes[label], pitch=pitch, is_drum=True, config=config)
                midi_files[label] = str(out_path)

            if not midi_files:
                notes.append("No drum MIDI files written due to low-confidence detection.")

        elif stem_type in {"bass", "vocals", "other"}:
            role_dir = output_dir_path / stem_type
            _ensure_dir(role_dir)

            audio_for_pitch = input_path
            if stem_type == "bass":
                audio_for_pitch = _preprocess_bass(input_path)

            min_freq, max_freq = (30.0, 300.0) if stem_type == "bass" else (80.0, 1200.0)
            midi_data, warnings = _predict_basic_pitch(
                audio_path=audio_for_pitch,
                min_freq=min_freq,
                max_freq=max_freq,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                min_note_length=min_note_length,
            )
            notes.extend(warnings)

            window_start, window_end = _grid_bounds(config)
            midi_notes = _clean_pitch_notes(
                midi_data,
                window_start=window_start,
                window_end=window_end,
                config=config,
                min_velocity=min_velocity,
                min_note_length=min_note_length,
                monophonic=(stem_type == "bass"),
            )

            if not midi_notes:
                notes.append(f"No {stem_type} notes survived filtering.")
            else:
                program = 32 if stem_type == "bass" else 52
                out_path = role_dir / f"{section_label}_{stem_type}.mid"
                _write_midi_notes(output_path=out_path, midi_notes=midi_notes, program=program, config=config)
                midi_files[stem_type] = str(out_path)

            if stem_type == "bass" and audio_for_pitch != input_path:
                try:
                    audio_for_pitch.unlink(missing_ok=True)
                except Exception as exc:
                    logger.warning("Failed to remove temporary bass file: %s", exc)

        else:
            raise ValueError("stem_type must be one of: drums, bass, vocals, other")

        basic_pitch_device = "not_used"
        if stem_type != "drums":
            basic_pitch_device = "gpu" if hardware.tensorflow_gpus else "cpu"

        model_info = {
            "drum_transcriber": "stft-energy-flux" if stem_type == "drums" else None,
            "pitch_transcriber": "basic_pitch" if stem_type != "drums" else None,
            "basic_pitch_model": "ICASSP_2022" if stem_type != "drums" else None,
            "hardware_used": {"basic_pitch": basic_pitch_device},
            "notes": "Tempo/grid authority enforced; notes snapped to grid steps.",
        }

        manifest_path = _write_manifest(
            output_dir=output_dir_path,
            hardware=hardware,
            model_info=model_info,
            midi_files=midi_files,
            notes=notes,
        )

        result = TranscriptionResult(
            stem_type=stem_type,
            output_dir=output_dir_path,
            midi_files=midi_files,
            manifest_path=manifest_path,
            notes=notes,
        )

        return {
            "input": str(input_path),
            "stem_type": result.stem_type,
            "output_dir": str(result.output_dir),
            "midi_files": result.midi_files,
            "manifest_path": result.manifest_path,
            "notes": result.notes,
        }
    except Exception as exc:
        trace = traceback.format_exc()
        logger.error("Transcription failed: %s", exc)
        logger.error(trace)
        return {
            "input": str(audio_path),
            "stem_type": stem_type,
            "output_dir": output_dir,
            "midi_files": {},
            "manifest_path": "",
            "notes": ["transcription_failed", str(exc), trace],
        }


###############################################################################
# MAIN
###############################################################################


def run() -> None:
    _configure_logging()
    mcp.run()


if __name__ == "__main__":
    run()