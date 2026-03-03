"""
uv run src/transcribe.py
"""
import io
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps
import torch
import time

ipa_model_id = "thewh1teagle/whisper-heb-ipa-large-v3-turbo-ct2"
text_model_id = "ivrit-ai/whisper-large-v3-turbo-ct2"
device = "cuda"
compute_type = "int8"
language = "he"
max_chunk_s = 25
sr = 16000

ipa_model = WhisperModel(ipa_model_id, device=device, compute_type=compute_type)
text_model = WhisperModel(text_model_id, device=device, compute_type=compute_type)
vad_model = load_silero_vad()

array, sampling_rate = sf.read("433_2120_10152_10474.wav")
if sampling_rate != sr:
    import librosa
    array = librosa.resample(array, orig_sr=sampling_rate, target_sr=sr)
audio = array.astype(np.float32)

wav_tensor = torch.from_numpy(audio)
start = time.time()
timestamps = get_speech_timestamps(wav_tensor, vad_model, return_seconds=True, sampling_rate=sr)

chunks = []
for ts in timestamps:
    chunk_start = int(ts["start"] * sr)
    chunk_end = int(ts["end"] * sr)
    max_samples = max_chunk_s * sr
    while chunk_end - chunk_start > max_samples:
        chunks.append((chunk_start, chunk_start + max_samples))
        chunk_start += max_samples
    if chunk_end > chunk_start:
        chunks.append((chunk_start, chunk_end))

merged = []
current_start, current_end = chunks[0]
for chunk_start, chunk_end in chunks[1:]:
    if (chunk_end - current_start) <= max_chunk_s * sr:
        current_end = chunk_end
    else:
        merged.append((current_start, current_end))
        current_start, current_end = chunk_start, chunk_end
merged.append((current_start, current_end))

print(f"Chunks before merge: {len(chunks)} | after merge: {len(merged)}")
print()

for chunk_start, chunk_end in merged:
    chunk = audio[chunk_start:chunk_end]
    offset = chunk_start / sr

    text_segs, _ = text_model.transcribe(chunk, beam_size=5, language=language, temperature=0, condition_on_previous_text=False)
    ipa_segs, _ = ipa_model.transcribe(chunk, beam_size=5, language=language, temperature=0, condition_on_previous_text=False, no_speech_threshold=0.1)

    text_out = " ".join(s.text.strip() for s in text_segs)
    ipa_out = " ".join(s.text.strip() for s in ipa_segs)

    if text_out or ipa_out:
        print(f"[{offset:.2f}s]\t{text_out}\t{ipa_out}")

end = time.time()
total_audio_s = len(audio) / sr
elapsed_s = end - start
rtf = elapsed_s / total_audio_s if total_audio_s > 0 else 0
print(f"\nElapsed: {elapsed_s:.2f}s | Audio: {total_audio_s:.2f}s | RTF: {rtf:.3f}")