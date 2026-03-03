"""
uv run src/download.py
"""

import os
import io
import soundfile as sf
import pandas as pd
from datasets import load_dataset, Audio

ds = load_dataset("yanirmr/VoxKnesset", split="train", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

transcripts = pd.read_parquet("hf://datasets/yanirmr/voxknesset/transcripts.parquet")
text_lookup = dict(zip(transcripts["filename"], transcripts["text"]))

sample = next(iter(ds))
filename = os.path.basename(sample["audio"]["path"])
text = text_lookup.get(filename, "N/A")

array, sampling_rate = sf.read(io.BytesIO(sample["audio"]["bytes"]))

print(f"File:    {filename}")
print(f"Speaker: {sample['speaker_id']}  Age: {sample['age']:.1f}  Gender: {'M' if sample['gender'] == 1 else 'F'}")
print(f"Text:    {text[:120]}")
print(f"Audio:   {len(array)} samples @ {sampling_rate} Hz")

sf.write(filename, array, sampling_rate)
print(f"Saved:   {filename}")