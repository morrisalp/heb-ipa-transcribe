# heb-ipa-transcribe

IPA heb transcribe pipeline

## Download

```console
uv run src/download.py
```

Audio is saved to `./dataset_output/audio/`.

## Transcribe

```console
uv run src/transcribe.py [OPTIONS]
```

Reads `.wav` files from `<input-dir>/audio/`, writes `metadata_ipa.csv` and `metadata_chunks.csv` to `<input-dir>/`, and pushes `(file_id, transcript)` rows to HuggingFace by default. Re-runs overwrite previous outputs.

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir DIR` | `./dataset_output` | Root dir; audio read from `<input-dir>/audio/` |
| `--chunks-dir DIR` | `./ipa_voxknesset` | Directory to write VAD-split audio chunks |
| `--batch-size N` | `8` | Chunks per `model.generate()` call |
| `--workers-per-gpu N` | `1` | Worker processes per GPU |
| `--hf-dataset REPO` | `malper/knesset-vox-vocalized` | HuggingFace dataset to push results to |
| `--no-push` | — | Skip pushing results to HuggingFace |
| `--dry-run` | — | Process only 5 files and push |
