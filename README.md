# heb-ipa-transcribe

IPA heb transcribe pipeline

## Download

```console
uv run src/download.py
```

## Transcribe

```console
uv run src/transcribe.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir DIR` | `./dataset_output` | Directory containing `.wav` files to transcribe |
| `--chunks-dir DIR` | `./ipa_voxknesset` | Directory to write VAD-split audio chunks |
| `--batch-size N` | `8` | Chunks per `model.generate()` call |
| `--workers-per-gpu N` | `1` | Worker processes per GPU |

Outputs `metadata_ipa.csv` (per-file transcripts) and `metadata_chunks.csv` (per-chunk transcripts) inside `--input-dir`. These files are gitignored.
