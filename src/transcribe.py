#!/usr/bin/env python3
"""
Audio IPA Transcriber - MULTI GPU / MULTI WORKER
Uses malper/abjadsr-he-finetune: outputs word-aligned hebrew=ascii_ipa pairs.
uv run src/transcribe.py
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import torch
import librosa
import soundfile as sf
from datetime import datetime
import multiprocessing
import queue
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_ID = "malper/abjadsr-he-finetune"
# Processor not uploaded to fine-tune repo — confirmed same arch as openai/whisper-large-v3-turbo
PROCESSOR_ID = "openai/whisper-large-v3-turbo"
SR = 16000
MAX_CHUNK_S = 25
BATCH_SIZE = 8  # chunks per model.generate() call


def parse_output(output: str) -> Tuple[str, str]:
    """Split 'word=ipa ...' into (hebrew_text, ascii_ipa)."""
    pairs = [token.split("=", 1) for token in output.split() if "=" in token]
    return " ".join(p[0] for p in pairs), " ".join(p[1] for p in pairs)


def gpu_worker(worker_id: int, target_gpu: int, input_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue, chunks_dir: str):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(target_gpu)

        print(f"Worker {worker_id} starting on Physical GPU {target_gpu}...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        processor = WhisperProcessor.from_pretrained(PROCESSOR_ID)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype)
        model = model.to(device)
        model.eval()

        forced_ids = processor.get_decoder_prompt_ids(language="he", task="transcribe")
        vad_model = load_silero_vad()

        print(f"Worker {worker_id} Ready!")

        while True:
            task = input_queue.get()
            if task is None:
                break

            audio_path_str, relative_path = task

            try:
                # 1. Load & resample
                audio, sr_orig = sf.read(audio_path_str, dtype='float32', always_2d=False)
                if sr_orig != SR:
                    audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=SR)

                # 2. VAD & merge into chunks <= MAX_CHUNK_S seconds
                timestamps = get_speech_timestamps(
                    torch.from_numpy(audio), vad_model, return_seconds=True, sampling_rate=SR
                )

                if not timestamps:
                    result_queue.put({
                        'filename': relative_path,
                        'text': "",
                        'phonemes': "",
                        'processed_at': datetime.now().isoformat()
                    })
                    continue

                max_samples = MAX_CHUNK_S * SR
                merged_chunks = []
                current_start = int(timestamps[0]["start"] * SR)
                current_end = int(timestamps[0]["end"] * SR)

                for ts in timestamps[1:]:
                    chunk_start = int(ts["start"] * SR)
                    chunk_end = int(ts["end"] * SR)
                    if (chunk_end - current_start) <= max_samples:
                        current_end = chunk_end
                    else:
                        merged_chunks.append(audio[current_start:current_end])
                        current_start, current_end = chunk_start, chunk_end
                merged_chunks.append(audio[current_start:current_end])

                # 3. Save chunk wavs
                relative_dir = os.path.dirname(relative_path)
                base_name = os.path.splitext(os.path.basename(relative_path))[0]
                target_dir = os.path.join(chunks_dir, relative_dir)
                os.makedirs(target_dir, exist_ok=True)

                for i, chunk in enumerate(merged_chunks):
                    sf.write(os.path.join(target_dir, f"{base_name}_{i:03d}.wav"), chunk, SR)

                # 4. Batched transcription
                full_text: List[str] = []
                full_ipa: List[str] = []
                chunk_metadata: List[Dict] = []

                for batch_start in range(0, len(merged_chunks), BATCH_SIZE):
                    batch = merged_chunks[batch_start:batch_start + BATCH_SIZE]

                    inputs = processor(
                        batch,
                        sampling_rate=SR,
                        return_tensors="pt",
                        padding=True,
                    )
                    input_features = inputs.input_features.to(device, dtype=dtype)

                    with torch.no_grad():
                        generated = model.generate(
                            input_features,
                            forced_decoder_ids=forced_ids,
                            max_new_tokens=444,
                        )

                    decoded = processor.batch_decode(generated, skip_special_tokens=True)

                    for i, raw in enumerate(decoded):
                        chunk_idx = batch_start + i
                        t_out, i_out = parse_output(raw.strip())
                        if t_out:
                            full_text.append(t_out)
                        if i_out:
                            full_ipa.append(i_out)
                        chunk_metadata.append({
                            "filename": os.path.join(relative_dir, f"{base_name}_{chunk_idx:03d}.wav"),
                            "text": t_out,
                            "phonemes": i_out,
                        })

                result_queue.put({
                    'filename': relative_path,
                    'text': " ".join(full_text),
                    'phonemes': " ".join(full_ipa),
                    'chunks': chunk_metadata,
                    'processed_at': datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Error processing {relative_path}: {e}")
                result_queue.put({'error': str(e), 'filename': relative_path})

    except Exception as e:
        logger.critical(f"Worker {worker_id} crashed: {e}")
    finally:
        print(f"Worker {worker_id} finished.")


class Transcriber:
    def __init__(self, input_dir: str, chunks_dir: str, workers_per_gpu: int = 1):
        self.input_dir = Path(input_dir)
        self.chunks_dir = Path(chunks_dir)
        self.output_csv = self.input_dir / "metadata_ipa.csv"
        self.chunks_csv = self.input_dir / "metadata_chunks.csv"
        self.checkpoint_file = self.input_dir / "checkpoint_ipa.json"

        try:
            self.num_gpus = torch.cuda.device_count()
        except Exception:
            self.num_gpus = 1

        if self.num_gpus == 0:
            logger.warning("No GPUs detected, falling back to CPU (will be slow)")
            self.num_gpus = 1

        self.workers_per_gpu = workers_per_gpu
        self.total_workers = self.num_gpus * self.workers_per_gpu
        self.save_interval = 20

    def load_checkpoint(self) -> Dict:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"processed": {}}

    def save_checkpoint(self, checkpoint: Dict):
        checkpoint["last_updated"] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    def get_pending_tasks(self, processed_files: set) -> List[Tuple[str, str]]:
        tasks = []
        for f in self.input_dir.glob("**/*.wav"):
            rel_path = str(f.relative_to(self.input_dir))
            if rel_path not in processed_files:
                tasks.append((str(f), rel_path))
        return tasks

    def append_chunks_csv(self, chunks_data: List[Dict]):
        if not chunks_data:
            return
        df = pd.DataFrame(chunks_data)
        header = not self.chunks_csv.exists()
        df.to_csv(self.chunks_csv, mode='a', header=header, index=False)

    def export_csv(self, checkpoint: Dict):
        results = list(checkpoint["processed"].values())
        if results:
            for r in results:
                if 'text' not in r:
                    r['text'] = ''
            df = pd.DataFrame(results)[['filename', 'text', 'phonemes']]
            df.to_csv(self.output_csv, index=False)

    def process_batch(self, tasks: List[Tuple[str, str]], checkpoint: Dict):
        manager = multiprocessing.Manager()
        input_queue = manager.Queue()
        result_queue = manager.Queue()

        for t in tasks:
            input_queue.put(t)
        for _ in range(self.total_workers):
            input_queue.put(None)

        workers = []
        worker_id = 0
        for gpu_id in range(self.num_gpus):
            for _ in range(self.workers_per_gpu):
                p = multiprocessing.Process(
                    target=gpu_worker,
                    args=(worker_id, gpu_id, input_queue, result_queue, str(self.chunks_dir))
                )
                p.start()
                workers.append(p)
                worker_id += 1

        pbar = tqdm(total=len(tasks), desc="Processing", unit="file")
        completed = 0

        while completed < len(tasks):
            try:
                res = result_queue.get(timeout=5)
                if 'error' not in res:
                    checkpoint["processed"][res['filename']] = {
                        "filename": res['filename'],
                        "text": res.get('text', ''),
                        "phonemes": res.get('phonemes', '')
                    }
                    if 'chunks' in res:
                        self.append_chunks_csv(res['chunks'])
                completed += 1
                pbar.update(1)

                if completed % self.save_interval == 0:
                    self.save_checkpoint(checkpoint)
                    self.export_csv(checkpoint)
            except queue.Empty:
                if all(not p.is_alive() for p in workers):
                    break

        pbar.close()
        for p in workers:
            p.join()

        self.save_checkpoint(checkpoint)
        self.export_csv(checkpoint)

    def run(self):
        print(f"Scanning {self.input_dir} for .wav files...")
        print(f"Config: {self.num_gpus} GPU(s), {self.workers_per_gpu} worker(s) per GPU ({self.total_workers} total).")

        checkpoint = self.load_checkpoint()
        tasks = self.get_pending_tasks(set(checkpoint["processed"].keys()))

        if not tasks:
            print("No new files to process. Exiting.")
            return

        print(f"Found {len(tasks)} new files. Starting processing batch...")
        self.process_batch(tasks, checkpoint)
        print("Processing complete.")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    TARGET_DIR = "./dataset_output"
    CHUNKS_DIR = "./ipa_voxknesset"
    os.makedirs(TARGET_DIR, exist_ok=True)
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    # 1 worker per GPU — HF model is larger than CTranslate2; increase if VRAM allows
    generator = Transcriber(input_dir=TARGET_DIR, chunks_dir=CHUNKS_DIR, workers_per_gpu=1)

    try:
        generator.run()
    except KeyboardInterrupt:
        print("\nProcess stopped by user.")
