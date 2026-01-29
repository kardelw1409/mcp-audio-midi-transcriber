# mcp-audio-midi-transcriber

GPU-first MCP server (`stdio`) for audio -> MIDI transcription intended for REAPER pipelines.

Run:

```powershell
py -3.11 -m mcp_audio_midi_transcriber
```

Notes:
- GPU: CUDA availability is detected via torch, and TensorFlow GPU devices are logged for Basic Pitch.
- Basic Pitch is used for pitch-focused stems (bass, vocals, other). Drums use fast STFT energy-flux detection.
- Drum onsets are quantized relative to `grid_start_seconds` (grid anchor) so MIDI aligns with tempo/tacts while preserving initial silence.
- `tempo` and `grid_start_seconds` are **required** to enforce grid alignment.
- Each run writes `transcription_manifest.json` with hardware/model metadata and confidence notes.

Tool:
- `transcribe_audio(audio_path, stem_type, output_dir=..., tempo=..., grid_start_seconds=..., start_bar=1, end_bar=None, bars=None, steps_per_bar=16, section_name=None, drums_fast_mode=True, ...)`

Example:

```json
{
  "audio_path": "C:/audio/bass.wav",
  "stem_type": "bass",
  "tempo": 172.2656,
  "grid_start_seconds": 12.2369,
  "bars": {"start_bar": 1, "end_bar": 16},
  "output_dir": "midi_transcribed"
}
```

Model selection matrix:

| Source | Best Method | Why |
| ------ | ----------- | --- |
| Kick | Onset + filter | ML overkill, timing critical |
| Snare | Transient detect | Clear transient energy |
| Hats | Energy clusters | Pitch irrelevant, rely on highs |
| Bass | Monophonic pitch | Rhythm > pitch density |
| Vocals | Melody extraction | Contour only |
