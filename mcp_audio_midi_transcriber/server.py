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


def _hard_quantize(t: float, grid: float) -> float:
    return round(t / grid) * grid


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
    return _hard_quantize(time_value, step)


def _quantize_time_relative(time_value: float, config: GridConfig) -> float:
    step = _grid_step_seconds(config)
    relative = time_value - config.grid_start_seconds
    snapped = _hard_quantize(relative, step)
    return config.grid_start_seconds + snapped


def _quantize_duration(duration: float, config: GridConfig, min_seconds: float) -> float:
    step = _grid_step_seconds(config)
    snapped = max(min_seconds, _hard_quantize(duration, step))
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
        step = _grid_step_seconds(config)
        for onset_time in onsets:
            start = max(0.0, float(onset_time) + float(offset))
            end = start + 0.08
            clipped = _clip_to_window(start, end, window_start, window_end)
            if clipped is None:
                continue
            start, end = clipped
            start = _hard_quantize(start, step)
            end = _hard_quantize(end, step)
            if end <= start:
                end = start + step
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
        step = _grid_step_seconds(config)
        start = _hard_quantize(start, step)
        end = _hard_quantize(end, step)
        if end <= start:
            end = start + step
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
    tempo_bpm: float,
) -> tuple[Any, list[str]]:
    notes: list[str] = []

    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict

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
    except Exception as exc:
        notes.append(f"Basic Pitch unavailable; using librosa.pyin fallback: {exc}")
        return _fallback_pitch_transcription(
            audio_path=audio_path,
            min_freq=min_freq,
            max_freq=max_freq,
            min_note_length=min_note_length,
            tempo_bpm=tempo_bpm,
            notes=notes,
        )


def _fallback_pitch_transcription(
    *,
    audio_path: Path,
    min_freq: float,
    max_freq: float,
    min_note_length: float,
    tempo_bpm: float,
    notes: list[str],
) -> tuple[Any, list[str]]:
    import numpy as np
    import librosa
    import pretty_midi

    hop_length = 512
    y, sr = librosa.load(path=str(audio_path), mono=True, sr=22050)
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=max(20.0, float(min_freq)),
        fmax=max(float(min_freq), float(max_freq)),
        hop_length=hop_length,
        sr=sr,
    )
    frame_times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    midi_values = librosa.hz_to_midi(f0)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length, center=True)[0]

    if len(rms) < len(midi_values):
        pad_value = float(rms[-1]) if len(rms) else 0.0
        rms = np.pad(rms, (0, len(midi_values) - len(rms)), mode="constant", constant_values=pad_value)
    elif len(rms) > len(midi_values):
        rms = rms[: len(midi_values)]

    def _median_smooth(arr: Any, k: int = 5) -> list[float | None]:
        result: list[float | None] = []
        half = k // 2
        for index in range(len(arr)):
            window = arr[max(0, index - half) : min(len(arr), index + half + 1)]
            window = [value for value in window if value is not None and np.isfinite(value)]
            result.append(float(np.median(window)) if window else None)
        return result

    def _apply_hysteresis(midi: list[float | None], threshold: float = 0.5) -> list[float | None]:
        stable: list[float | None] = []
        last: float | None = None

        for pitch in midi:
            if pitch is None:
                stable.append(None)
                continue

            if last is None:
                stable.append(float(pitch))
                last = float(pitch)
                continue

            if abs(float(pitch) - last) < threshold:
                stable.append(last)
            else:
                stable.append(float(pitch))
                last = float(pitch)

        return stable

    smoothed_midi = _median_smooth(midi_values, k=5)
    stable_midi = _apply_hysteresis(smoothed_midi, threshold=0.5)

    for index, voiced in enumerate(voiced_flag):
        if not voiced:
            stable_midi[index] = None

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
    instrument = pretty_midi.Instrument(program=0, is_drum=False)

    segment_frames: list[int] = []
    segment_pitches: list[float] = []
    segment_energies: list[float] = []
    frame_hop_seconds = hop_length / sr

    def _flush_segment() -> None:
        if len(segment_frames) < 3:
            segment_frames.clear()
            segment_pitches.clear()
            segment_energies.clear()
            return

        start_idx = segment_frames[0]
        end_idx = segment_frames[-1]
        start_time = float(frame_times[start_idx])
        end_time = float(frame_times[end_idx] + frame_hop_seconds)
        if end_time <= start_time:
            end_time = start_time + frame_hop_seconds
        if end_time - start_time < min_note_length:
            segment_frames.clear()
            segment_pitches.clear()
            segment_energies.clear()
            return

        average_pitch = float(np.mean(segment_pitches))
        if not np.isfinite(average_pitch):
            segment_frames.clear()
            segment_pitches.clear()
            segment_energies.clear()
            return

        velocity = int(np.clip(float(np.mean(segment_energies)) * 127.0 * 3.0, 30, 110))
        instrument.notes.append(
            pretty_midi.Note(
                velocity=velocity,
                pitch=int(round(average_pitch)),
                start=start_time,
                end=end_time,
            )
        )
        segment_frames.clear()
        segment_pitches.clear()
        segment_energies.clear()

    for index, pitch in enumerate(stable_midi):
        if pitch is None or not np.isfinite(pitch):
            _flush_segment()
            continue

        if segment_pitches and abs(float(pitch) - float(segment_pitches[-1])) >= 1.0:
            _flush_segment()

        segment_frames.append(index)
        segment_pitches.append(float(pitch))
        segment_energies.append(float(rms[index]))

    _flush_segment()

    midi.instruments.append(instrument)
    notes.append("Used librosa.pyin fallback because Basic Pitch was unavailable.")
    return midi, notes


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
            step = _grid_step_seconds(config)
            start = _hard_quantize(start, step)
            end = _hard_quantize(end, step)
            if end <= start:
                end = start + step
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


@mcp.tool("transcribe_audio")
def transcribe_audio(audio_path: str, stem_type: StemType, output_dir: str = "midi_transcribed", tempo: float | None = None, grid_start_seconds: float | None = None, start_bar: int = 1, end_bar: int | None = None, bars: dict[str, int] | None = None, steps_per_bar: int = 16, section_name: str | None = None, onset_threshold: float = 0.1, frame_threshold: float = 0.3, min_note_length: float = 0.05, min_velocity: int = 28, drums_fast_mode: bool = True) -> dict[str, Any]:
    return transcribe_audio_core(
        audio_path=audio_path,
        stem_type=stem_type,
        output_dir=output_dir,
        tempo=tempo,
        grid_start_seconds=grid_start_seconds,
        start_bar=start_bar,
        end_bar=end_bar,
        bars=bars,
        steps_per_bar=steps_per_bar,
        section_name=section_name,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        min_note_length=min_note_length,
        min_velocity=min_velocity,
        drums_fast_mode=drums_fast_mode,
    )


def transcribe_audio_core(
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
        min_note_length = _grid_step_seconds(config)

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
                tempo_bpm=tempo,
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

        result = {
            "input": str(input_path),
            "stem_type": result.stem_type,
            "output_dir": str(result.output_dir),
            "midi_files": result.midi_files,
            "manifest_path": result.manifest_path,
            "notes": result.notes,
        }
        editable_marker = os.environ.get("MCP_WORKFLOW_EDITABLE_TEST")
        if editable_marker:
            result["editable_install_marker"] = editable_marker
        return result
    except Exception as exc:
        trace = traceback.format_exc()
        logger.error("Transcription failed: %s", exc)
        logger.error(trace)
        result = {
            "input": str(audio_path),
            "stem_type": stem_type,
            "output_dir": output_dir,
            "midi_files": {},
            "manifest_path": "",
            "notes": ["transcription_failed", str(exc), trace],
        }
        editable_marker = os.environ.get("MCP_WORKFLOW_EDITABLE_TEST")
        if editable_marker:
            result["editable_install_marker"] = editable_marker
        return result


###############################################################################
# MIDI Refinement Engine
###############################################################################


@mcp.tool("refine_midi_engine")
def refine_midi_engine(
    midi_path: str,
    mode: str = "melody",
    variation: float = 0.3,
    preserve_intent: float = 0.8,
    num_variations: int = 5,
    magenta_variations: int = 2,
    key: str = "auto",
    quantize_strength: float = 0.8,
    strict_pitch: bool = False,
    output_dir: str | None = None,
) -> dict[str, Any]:
    return refine_midi_engine_core(
        midi_path=midi_path,
        mode=mode,
        variation=variation,
        preserve_intent=preserve_intent,
        num_variations=num_variations,
        magenta_variations=magenta_variations,
        key=key,
        quantize_strength=quantize_strength,
        strict_pitch=strict_pitch,
        output_dir=output_dir,
    )


def refine_midi_engine_core(
    midi_path: str,
    mode: str = "melody",
    variation: float = 0.3,
    preserve_intent: float = 0.8,
    num_variations: int = 5,
    magenta_variations: int = 2,
    key: str = "auto",
    quantize_strength: float = 0.8,
    strict_pitch: bool = False,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """
    Transform a dirty MIDI file into multiple musically improved MIDI variations.
    Produces a clean refined base, deterministic variations, and Magenta-style variations.
    """
    _configure_logging()
    notes_log: list[str] = []

    try:
        import copy
        import pretty_midi
        import numpy as np

        # --- 1. Load MIDI ---
        midi_in_path = Path(midi_path).expanduser().resolve()
        if not midi_in_path.exists():
            raise FileNotFoundError(f"midi_path not found: {midi_in_path}")
        pm = pretty_midi.PrettyMIDI(str(midi_in_path))

        all_notes: list[Any] = []
        for instrument in pm.instruments:
            for note in instrument.notes:
                all_notes.append(note)
        all_notes.sort(key=lambda n: n.start)

        if not all_notes:
            raise ValueError("No notes found in MIDI file.")

        # --- 2. Analyze MIDI ---
        pitches = [n.pitch for n in all_notes]
        durations = [n.end - n.start for n in all_notes]
        pitch_min, pitch_max = min(pitches), max(pitches)
        end_time = pm.get_end_time()
        note_density = len(all_notes) / max(end_time, 1e-9)
        notes_log.append(
            f"Analysis: {len(all_notes)} notes, pitch range {pitch_min}-{pitch_max}, "
            f"density {note_density:.2f} notes/sec, avg_duration {sum(durations) / len(durations):.3f}s"
        )

        tempo_changes = pm.get_tempo_changes()
        tempo_bpm = float(tempo_changes[1][0]) if len(tempo_changes[1]) > 0 else 120.0
        beat_length = 60.0 / tempo_bpm
        grid = beat_length / 4.0  # sixteenth-note grid
        strict_pitch = bool(strict_pitch or mode in {"vocals", "melody"})

        # --- 3. Key Detection ---
        NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
        MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]

        def _build_pitch_histogram(note_list: list[Any]) -> np.ndarray:
            hist = np.zeros(12)
            for n in note_list:
                hist[n.pitch % 12] += 1
            return hist

        def _detect_key_auto(note_list: list[Any]) -> str:
            hist = _build_pitch_histogram(note_list)
            hist_norm = hist / (hist.sum() + 1e-9)
            best_score = -1.0
            best_key = "C major"
            for root in range(12):
                for profile, suffix in [(MAJOR_PROFILE, "major"), (MINOR_PROFILE, "minor")]:
                    rotated = np.roll(profile, root)
                    norm = rotated / (rotated.sum() + 1e-9)
                    score = float(np.dot(hist_norm, norm))
                    if score > best_score:
                        best_score = score
                        best_key = f"{NOTE_NAMES[root]} {suffix}"
            return best_key

        def _get_scale_pitches(key_str: str) -> list[int]:
            parts = key_str.strip().split()
            root_name = parts[0] if parts else "C"
            mode_name = parts[1] if len(parts) > 1 else "major"
            root = NOTE_NAMES.index(root_name) if root_name in NOTE_NAMES else 0
            intervals = MAJOR_INTERVALS if mode_name == "major" else MINOR_INTERVALS
            result: list[int] = []
            for octave in range(11):
                for interval in intervals:
                    p = root + interval + 12 * octave
                    if 0 <= p <= 127:
                        result.append(p)
            return sorted(set(result))

        if strict_pitch:
            detected_key = "strict_pitch"
            scale_pitches: list[int] = []
            notes_log.append("Strict pitch mode enabled; skipping key detection and scale snapping.")
        else:
            detected_key = _detect_key_auto(all_notes) if key == "auto" else key
            notes_log.append(f"{'Detected' if key == 'auto' else 'Using provided'} key: {detected_key}")
            scale_pitches = _get_scale_pitches(detected_key)

        # --- 4. Scale Snapping ---
        def _snap_to_scale(pitch: int, scale: list[int]) -> int:
            if not scale:
                return pitch
            return min(scale, key=lambda p: (abs(p - pitch), p))

        def _scale_idx(pitch: int, scale: list[int]) -> int:
            snapped = _snap_to_scale(pitch, scale)
            try:
                return scale.index(snapped)
            except ValueError:
                return 0

        def _snap_to_scale_directional(pitch: int, scale: list[int], prev_pitch: int | None) -> int:
            """Snap to nearest scale pitch while respecting melodic direction."""
            if not scale or prev_pitch is None:
                return _snap_to_scale(pitch, scale)
            direction = pitch - prev_pitch
            if direction > 0:
                candidates = [p for p in scale if p >= prev_pitch]
            elif direction < 0:
                candidates = [p for p in scale if p <= prev_pitch]
            else:
                candidates = scale
            if not candidates:
                candidates = scale
            snapped = min(candidates, key=lambda p: abs(p - pitch))
            # fall back to nearest if directional snap introduces a large jump
            if abs(snapped - pitch) > 12 and abs(direction) <= 12:
                snapped = _snap_to_scale(pitch, scale)
            return snapped

        # --- 5. Preserve Melodic Contour ---
        def _apply_contour(
            snapped_list: list[int],
            orig_list: list[int],
            scale: list[int],
            strength: float,
        ) -> list[int]:
            if len(snapped_list) < 2:
                return list(snapped_list)
            result = [snapped_list[0]]
            for i in range(1, len(snapped_list)):
                interval = orig_list[i] - orig_list[i - 1]
                cur = result[-1]
                cur_idx = _scale_idx(cur, scale)
                # Map semitone interval to approximate scale steps (~2 semitones per diatonic step)
                scale_step = (
                    round(interval / 2.0)
                    if abs(interval) >= 2
                    else (1 if interval > 0 else (-1 if interval < 0 else 0))
                )
                target_idx = max(0, min(len(scale) - 1, cur_idx + scale_step))
                contour_pitch = scale[target_idx]
                # high strength → preserve computed contour; low → allow nearest-pitch snap
                blended = int(round(strength * contour_pitch + (1.0 - strength) * snapped_list[i]))
                result.append(_snap_to_scale(blended, scale))
            return result

        # --- 6. Rhythm Quantization ---
        def _quantize_t(t: float, g: float, strength: float) -> float:
            _ = strength
            return _hard_quantize(t, g)

        # --- 11. Cleanup (defined early; used throughout) ---
        def _cleanup(note_list: list[Any], min_dur: float = 0.05) -> list[Any]:
            cleaned = sorted(note_list, key=lambda n: (n.start, n.pitch))
            result: list[Any] = []
            last_end: dict[int, float] = {}
            for note in cleaned:
                if note.end - note.start < min_dur:
                    continue
                p = note.pitch
                if last_end.get(p, -1.0) > note.start:
                    note = copy.copy(note)
                    note.start = last_end[p]
                    if note.end - note.start < min_dur:
                        continue
                note = copy.copy(note)
                note.velocity = _normalize_velocity(note.velocity, min_velocity=1)
                result.append(note)
                last_end[p] = note.end
            return result

        # --- 6b. Humanization ---
        _hum_rng = np.random.default_rng(17)

        def _humanize(note_list: list[Any]) -> list[Any]:
            """Apply velocity curves and micro-timing drift for a human feel."""
            result: list[Any] = []
            n_notes = len(note_list)
            for i, note in enumerate(note_list):
                n = copy.copy(note)
                # Sinusoidal accent over the phrase (arch shape)
                phrase_pos = i / max(n_notes - 1, 1)
                accent = int(round(8.0 * np.sin(np.pi * phrase_pos)))
                rand_vel = int(_hum_rng.integers(-5, 6))
                n.velocity = _normalize_velocity(n.velocity + accent + rand_vel, min_velocity=1)
                # Smooth micro-timing: small Gaussian drift scaled by (1 - preserve_intent)
                drift = float(_hum_rng.normal(0.0, 0.005 * (1.0 - preserve_intent)))
                n.start = max(0.0, n.start + drift)
                n.end = max(n.start + 0.05, n.end + drift)
                result.append(n)
            return result

        # --- 7. Mode Logic helpers ---
        _key_parts = detected_key.split()
        _key_root = NOTE_NAMES.index(_key_parts[0]) if _key_parts and _key_parts[0] in NOTE_NAMES else 0
        _key_mode_str = _key_parts[1] if len(_key_parts) > 1 else "major"

        def _apply_bass_mode(note: Any) -> Any:
            target = note.pitch
            while target > 48:
                target -= 12
            while target < 36:
                target += 12
            snapped = _snap_to_scale(target, scale_pitches)
            # Root-note bias: nudge toward key root if we're close
            if abs((snapped % 12) - _key_root) <= 2:
                root_cands = [p for p in scale_pitches if p % 12 == _key_root and 36 <= p <= 48]
                if root_cands:
                    snapped = min(root_cands, key=lambda p: abs(p - target))
            note.pitch = snapped
            note.start = _hard_quantize(note.start, grid)
            # Longer note durations for bass — at least 1.5 beats
            dur = max(beat_length * 1.5, round((note.end - note.start) / grid) * grid)
            note.end = _hard_quantize(note.start + dur, grid)
            if note.end <= note.start:
                note.end = note.start + grid
            return note

        def _expand_chords(note_list: list[Any], scale: list[int]) -> list[Any]:
            expanded: list[Any] = []
            _chord_rng = np.random.default_rng(sum(n.pitch for n in note_list) if note_list else 0)
            for n in note_list:
                expanded.append(n)
                # Determine third: major (+4) if pitch+4 is already in scale, else minor (+3)
                third_semi = 4 if _snap_to_scale(n.pitch + 4, scale) == n.pitch + 4 else 3
                fifth_semi = 7
                # Occasional second inversion: 5th below root instead of 5th above
                invert = _chord_rng.random() < 0.2
                for semitones in (third_semi, fifth_semi):
                    en = copy.copy(n)
                    if invert and semitones == fifth_semi:
                        en.pitch = _snap_to_scale(n.pitch - 5, scale)
                    else:
                        en.pitch = _snap_to_scale(n.pitch + semitones, scale)
                    expanded.append(en)
            return expanded

        # --- 8. Build refined notes ---
        orig_pitches = [n.pitch for n in all_notes]
        if strict_pitch or not scale_pitches:
            contoured = orig_pitches
        else:
            # Context-aware directional snapping
            snapped: list[int] = []
            for _si, _p in enumerate(orig_pitches):
                _prev = snapped[_si - 1] if _si > 0 else None
                snapped.append(_snap_to_scale_directional(_p, scale_pitches, _prev))
            contoured = _apply_contour(snapped, orig_pitches, scale_pitches, preserve_intent)

        refined_notes: list[Any] = []
        for i, note in enumerate(all_notes):
            n = copy.copy(note)
            n.pitch = int(round(note.pitch if strict_pitch else contoured[i]))
            n.start = _hard_quantize(n.start, grid)
            n.end = _hard_quantize(n.end, grid)
            if n.end <= n.start:
                n.end = n.start + grid
            n.velocity = _normalize_velocity(n.velocity, min_velocity=1)
            if not strict_pitch and mode == "bass":
                n = _apply_bass_mode(n)
            refined_notes.append(n)

        refined_notes = _cleanup(refined_notes)
        if not strict_pitch and scale_pitches and mode == "chords":
            refined_notes = _cleanup(_expand_chords(refined_notes, scale_pitches))

        # --- 12. Output directory ---
        out_path = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else Path("midi_refined").resolve()
        )
        _ensure_dir(out_path)
        stem_name = midi_in_path.stem
        midi_files_out: list[str] = []

        def _write_notes_to_midi(note_list: list[Any], file_path: Path, bpm: float) -> None:
            program_num = 33 if mode == "bass" else 0
            pm_out = pretty_midi.PrettyMIDI(initial_tempo=bpm)
            instr = pretty_midi.Instrument(program=program_num)
            for n in note_list:
                start = float(max(0.0, n.start))
                end = float(max(start + 0.01, n.end))
                instr.notes.append(
                    pretty_midi.Note(
                        velocity=_normalize_velocity(n.velocity, min_velocity=1),
                        pitch=int(max(0, min(127, n.pitch))),
                        start=start,
                        end=end,
                    )
                )
            pm_out.instruments.append(instr)
            pm_out.write(str(file_path))

        # Save CLEAN MIDI (subtle humanization applied; refined_notes kept pure for variations)
        clean_path = out_path / f"{stem_name}_clean.mid"
        _write_notes_to_midi(_cleanup(list(refined_notes)), clean_path, tempo_bpm)
        midi_files_out.append(str(clean_path))
        notes_log.append(f"Saved clean MIDI: {clean_path.name}")

        # --- 9. Deterministic Variations ---
        _VAR_TYPES = ["phrase_shift", "rhythm", "density", "velocity_phrasing", "combined"]

        def _make_phrase_shift(base: list[Any], shift_steps: int) -> list[Any]:
            result: list[Any] = []
            for note in base:
                n = copy.copy(note)
                idx = _scale_idx(n.pitch, scale_pitches)
                new_idx = max(0, min(len(scale_pitches) - 1, idx + shift_steps))
                n.pitch = scale_pitches[new_idx]
                result.append(n)
            return _cleanup(_humanize(result))

        def _make_rhythm_var(base: list[Any], rng_inst: Any) -> list[Any]:
            result: list[Any] = []
            for note in base:
                n = copy.copy(note)
                dur = n.end - n.start
                dur_factor = 1.0 + float(rng_inst.uniform(-0.2, 0.2)) * variation
                new_dur = max(grid * 0.5, dur * dur_factor)
                synco = float(rng_inst.choice([-1, 0, 0, 1])) * grid * 0.5 * variation
                n.start = max(0.0, n.start + synco)
                n.end = n.start + new_dur
                result.append(n)
            return _cleanup(_humanize(result))

        def _make_density_var(base: list[Any], rng_inst: Any) -> list[Any]:
            result: list[Any] = []
            for note in base:
                r = float(rng_inst.random())
                if r < variation * 0.25:
                    continue
                result.append(copy.copy(note))
                if r > 1.0 - variation * 0.15:
                    dup = copy.copy(note)
                    half = (note.end - note.start) * 0.5
                    dup.start = note.start + half
                    dup.end = note.end + half
                    idx = _scale_idx(dup.pitch, scale_pitches)
                    dup.pitch = scale_pitches[
                        max(0, min(len(scale_pitches) - 1, idx + int(rng_inst.choice([-1, 1]))))
                    ]
                    result.append(dup)
            if not result:
                result = [copy.copy(base[0])] if base else []
            return _cleanup(_humanize(result))

        def _make_velocity_phrasing(base: list[Any]) -> list[Any]:
            result: list[Any] = []
            n_notes = len(base)
            for i, note in enumerate(base):
                n = copy.copy(note)
                phrase_pos = i / max(n_notes - 1, 1)
                arch = int(round(20.0 * np.sin(np.pi * phrase_pos) * variation))
                n.velocity = _normalize_velocity(n.velocity + arch, min_velocity=1)
                result.append(n)
            return _cleanup(result)

        rng = np.random.default_rng(42)
        for var_idx in range(num_variations):
            if strict_pitch:
                var_notes = _cleanup([copy.copy(note) for note in refined_notes])
                label = f"strict_{var_idx}"
            else:
                var_type = _VAR_TYPES[var_idx % len(_VAR_TYPES)]
                var_rng = np.random.default_rng(42 + var_idx * 13)
                if var_type == "phrase_shift":
                    shift = int(var_rng.choice([-2, -1, 1, 2]))
                    var_notes = _make_phrase_shift(refined_notes, shift)
                    label = f"phrase_shift_{'p' if shift > 0 else 'm'}{abs(shift)}"
                elif var_type == "rhythm":
                    var_notes = _make_rhythm_var(refined_notes, var_rng)
                    label = "rhythm"
                elif var_type == "density":
                    var_notes = _make_density_var(refined_notes, var_rng)
                    label = "density"
                elif var_type == "velocity_phrasing":
                    var_notes = _make_velocity_phrasing(refined_notes)
                    label = "velocity_phrasing"
                else:
                    shift = int(var_rng.choice([-1, 1]))
                    var_notes = _make_phrase_shift(refined_notes, shift)
                    var_notes = _make_rhythm_var(var_notes, var_rng)
                    label = "combined"
                if mode == "chords":
                    var_notes = _cleanup(_expand_chords(var_notes, scale_pitches))
            var_path = out_path / f"{stem_name}_var_{label}.mid"
            if str(var_path) in midi_files_out:
                var_path = out_path / f"{stem_name}_var_{label}_{var_idx}.mid"
            _write_notes_to_midi(var_notes, var_path, tempo_bpm)
            midi_files_out.append(str(var_path))
        notes_log.append(f"Saved {num_variations} named deterministic variations.")

        # --- 10. Magenta Variations (always attempted) ---
        _MAG_TRANSFORMS = ["time_stretch", "pitch_transpose", "density_variation", "retrograde"]

        def _apply_magenta_transform(base: list[Any], transform: str, mag_rng_inst: Any) -> list[Any]:
            result: list[Any] = []
            if transform == "time_stretch":
                direction = 1 if mag_rng_inst.random() > 0.5 else -1
                factor = float(max(0.7, min(1.4, 1.0 + direction * float(mag_rng_inst.uniform(0.08, 0.20)))))
                for note in base:
                    n = copy.copy(note)
                    n.start = n.start * factor
                    n.end = n.end * factor
                    result.append(n)
            elif transform == "pitch_transpose":
                steps = int(mag_rng_inst.choice([-3, -2, -1, 1, 2, 3]))
                for note in base:
                    n = copy.copy(note)
                    idx = _scale_idx(n.pitch, scale_pitches)
                    n.pitch = scale_pitches[max(0, min(len(scale_pitches) - 1, idx + steps))]
                    result.append(n)
            elif transform == "density_variation":
                prev_end = 0.0
                for i, note in enumerate(base):
                    if float(mag_rng_inst.random()) < 0.20 * variation:
                        continue
                    n = copy.copy(note)
                    # Insert scale-passing note when gap is large enough
                    if i > 0 and result:
                        gap = n.start - prev_end
                        if gap > grid * 1.5 and float(mag_rng_inst.random()) < 0.30 * variation:
                            passing = copy.copy(n)
                            prev_idx = _scale_idx(result[-1].pitch, scale_pitches)
                            cur_idx = _scale_idx(n.pitch, scale_pitches)
                            mid_idx = (prev_idx + cur_idx) // 2
                            passing.pitch = scale_pitches[max(0, min(len(scale_pitches) - 1, mid_idx))]
                            passing.start = prev_end + grid * 0.5
                            passing.end = passing.start + grid
                            passing.velocity = _normalize_velocity(n.velocity - 12, min_velocity=1)
                            result.append(passing)
                    result.append(n)
                    prev_end = n.end
            elif transform == "retrograde":
                pitches_rev = [note.pitch for note in reversed(base)]
                for i, note in enumerate(base):
                    n = copy.copy(note)
                    n.pitch = pitches_rev[i]
                    result.append(n)
            else:
                result = [copy.copy(n) for n in base]
            return _cleanup(_humanize(result))

        magenta_ok = False
        try:
            import note_seq  # type: ignore

            magenta_ok = True
        except ImportError:
            notes_log.append(
                "WARNING: note_seq (Magenta) not installed. "
                "Generating structurally distinct fallback variations."
            )

        if strict_pitch:
            for mag_idx in range(magenta_variations):
                mag_notes_fb = _cleanup([copy.copy(note) for note in refined_notes])
                mag_path = out_path / f"{stem_name}_magenta_strict_{mag_idx}.mid"
                _write_notes_to_midi(mag_notes_fb, mag_path, tempo_bpm)
                midi_files_out.append(str(mag_path))
            notes_log.append(f"Saved {magenta_variations} strict-pitch Magenta placeholders.")
        elif magenta_ok:
            try:
                ns = note_seq.midi_file_to_note_sequence(str(clean_path))
                for mag_idx in range(magenta_variations):
                    mag_rng_inst = np.random.default_rng(7 + mag_idx * 31)
                    transform = _MAG_TRANSFORMS[mag_idx % len(_MAG_TRANSFORMS)]
                    mag_ns = copy.deepcopy(ns)
                    if transform == "time_stretch":
                        direction = 1 if mag_rng_inst.random() > 0.5 else -1
                        factor = float(max(0.7, min(1.4, 1.0 + direction * float(mag_rng_inst.uniform(0.08, 0.20)))))
                        for n in mag_ns.notes:
                            n.start_time = n.start_time * factor
                            n.end_time = n.end_time * factor
                        mag_ns.total_time = mag_ns.total_time * factor
                    elif transform == "pitch_transpose":
                        steps = int(mag_rng_inst.choice([-3, -2, -1, 1, 2, 3]))
                        for n in mag_ns.notes:
                            idx = _scale_idx(n.pitch, scale_pitches)
                            n.pitch = scale_pitches[max(0, min(len(scale_pitches) - 1, idx + steps))]
                    else:
                        # density_variation / retrograde: operate on note list
                        mag_notes_t = _apply_magenta_transform(refined_notes, transform, mag_rng_inst)
                        if mode == "chords":
                            mag_notes_t = _cleanup(_expand_chords(mag_notes_t, scale_pitches))
                        mag_path = out_path / f"{stem_name}_magenta_transform_{mag_idx}.mid"
                        _write_notes_to_midi(mag_notes_t, mag_path, tempo_bpm)
                        midi_files_out.append(str(mag_path))
                        continue
                    mag_path = out_path / f"{stem_name}_magenta_transform_{mag_idx}.mid"
                    note_seq.note_sequence_to_midi_file(mag_ns, str(mag_path))
                    midi_files_out.append(str(mag_path))
                notes_log.append(f"Saved {magenta_variations} Magenta transform variations.")
            except Exception as mag_exc:
                notes_log.append(
                    f"WARNING: Magenta processing failed: {mag_exc}. "
                    "Generating structurally distinct fallback variations."
                )
                magenta_ok = False

        if not strict_pitch and not magenta_ok:
            for mag_idx in range(magenta_variations):
                mag_rng_inst = np.random.default_rng(99 + mag_idx * 31)
                transform = _MAG_TRANSFORMS[mag_idx % len(_MAG_TRANSFORMS)]
                mag_notes_fb = _apply_magenta_transform(refined_notes, transform, mag_rng_inst)
                if mode == "chords":
                    mag_notes_fb = _cleanup(_expand_chords(mag_notes_fb, scale_pitches))
                mag_path = out_path / f"{stem_name}_magenta_fallback_{mag_idx}_{transform}.mid"
                _write_notes_to_midi(mag_notes_fb, mag_path, tempo_bpm)
                midi_files_out.append(str(mag_path))
            notes_log.append(
                f"Saved {magenta_variations} structurally distinct fallback Magenta-style variations "
                "(note_seq unavailable)."
            )

        return {
            "input": str(midi_in_path),
            "output_dir": str(out_path),
            "midi_files": midi_files_out,
            "detected_key": detected_key,
            "notes": notes_log,
        }

    except Exception as exc:
        trace = traceback.format_exc()
        logger.error("refine_midi_engine failed: %s", exc)
        logger.error(trace)
        return {
            "input": str(midi_path),
            "output_dir": str(output_dir or "midi_refined"),
            "midi_files": [],
            "detected_key": "",
            "notes": ["refine_midi_engine_failed", str(exc), trace],
        }


###############################################################################
# MAIN
###############################################################################


def run() -> None:
    _configure_logging()
    mcp.run()


if __name__ == "__main__":
    run()