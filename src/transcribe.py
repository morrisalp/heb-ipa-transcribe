#!/usr/bin/env python3
"""
Audio IPA Transcriber - MULTI GPU / MULTI WORKER
Uses malper/abjadsr-he: outputs Hebrew IPA transcription.
uv run src/transcribe.py [OPTIONS]
"""

import os
import json
import argparse
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
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_ID = "malper/abjadsr-he"
# Processor not uploaded to model repo — confirmed same arch as openai/whisper-large-v3-turbo
PROCESSOR_ID = "openai/whisper-large-v3-turbo"
HF_DATASET_ID = "malper/knesset-vox-ipa"
SR = 16000
MAX_CHUNK_S = 25
DEFAULT_BATCH_SIZE = 8  # chunks per model.generate() call
DRY_RUN_N = 5


def gpu_worker(worker_id: int, target_gpu: int, input_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue, chunks_dir: str, batch_size: int):
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

            audio_path_str, file_id = task

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
                        'file_id': file_id,
                        'transcript': "",
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
                target_dir = os.path.join(chunks_dir, file_id)
                os.makedirs(target_dir, exist_ok=True)

                for i, chunk in enumerate(merged_chunks):
                    sf.write(os.path.join(target_dir, f"{i:03d}.wav"), chunk, SR)

                # 4. Batched transcription
                chunk_transcripts: List[str] = []
                chunk_metadata: List[Dict] = []

                for batch_start in range(0, len(merged_chunks), batch_size):
                    batch = merged_chunks[batch_start:batch_start + batch_size]

                    inputs = processor(
                        batch,
                        sampling_rate=SR,
                        return_tensors="pt",
                        padding=True,
                    )
                    input_features = inputs.input_features.to(device, dtype=dtype)
                    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

                    with torch.no_grad():
                        generated = model.generate(
                            input_features,
                            attention_mask=attention_mask,
                            forced_decoder_ids=forced_ids,
                            max_new_tokens=444,
                        )

                    decoded = processor.batch_decode(generated, skip_special_tokens=True)

                    for i, raw in enumerate(decoded):
                        chunk_idx = batch_start + i
                        raw = raw.strip()
                        chunk_transcripts.append(raw)
                        chunk_metadata.append({
                            "file_id": file_id,
                            "chunk_idx": chunk_idx,
                            "transcript": raw,
                        })

                result_queue.put({
                    'file_id': file_id,
                    'transcript': " ".join(chunk_transcripts),
                    'chunks': chunk_metadata,
                    'processed_at': datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Error processing {file_id}: {e}")
                result_queue.put({'error': str(e), 'file_id': file_id})

    except Exception as e:
        logger.critical(f"Worker {worker_id} crashed: {e}")
    finally:
        print(f"Worker {worker_id} finished.")


class Transcriber:
    def __init__(self, input_dir: str, chunks_dir: str, workers_per_gpu: int = 1,
                 batch_size: int = DEFAULT_BATCH_SIZE, hf_dataset: str = HF_DATASET_ID,
                 no_push: bool = False, dry_run: bool = False):
        self.input_dir = Path(input_dir)
        self.audio_dir = self.input_dir / "audio"
        self.chunks_dir = Path(chunks_dir)
        self.output_csv = self.input_dir / "metadata_ipa.csv"
        self.chunks_csv = self.input_dir / "metadata_chunks.csv"
        self.checkpoint_file = self.input_dir / "checkpoint_ipa.json"
        self.batch_size = batch_size
        self.hf_dataset = hf_dataset
        self.no_push = no_push
        self.dry_run = dry_run

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
        self._chunks_csv_started = False

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

    def get_pending_tasks(self, processed_ids: set) -> List[Tuple[str, str]]:
        tasks = []
        for f in self.audio_dir.glob("*.wav"):
            file_id = f.stem
            if file_id not in processed_ids:
                tasks.append((str(f), file_id))
        return tasks

    def append_chunks_csv(self, chunks_data: List[Dict]):
        if not chunks_data:
            return
        df = pd.DataFrame(chunks_data)
        header = not self._chunks_csv_started
        df.to_csv(self.chunks_csv, mode='a', header=header, index=False)
        self._chunks_csv_started = True

    def export_csv(self, checkpoint: Dict):
        results = list(checkpoint["processed"].values())
        if results:
            df = pd.DataFrame(results)[['file_id', 'transcript']]
            df.to_csv(self.output_csv, index=False)

    def push_to_hub(self, checkpoint: Dict):
        results = [v for v in checkpoint["processed"].values()]
        if not results:
            print("Nothing to push.")
            return
        rows = [{"file_id": r["file_id"], "transcript": r["transcript"]} for r in results]
        ds = Dataset.from_list(rows)
        ds.push_to_hub(self.hf_dataset)
        print(f"Pushed {len(rows)} rows to {self.hf_dataset}.")

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
                    args=(worker_id, gpu_id, input_queue, result_queue, str(self.chunks_dir), self.batch_size)
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
                    checkpoint["processed"][res['file_id']] = {
                        "file_id": res['file_id'],
                        "transcript": res.get('transcript', ''),
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
        print(f"Scanning {self.audio_dir} for .wav files...")
        print(f"Config: {self.num_gpus} GPU(s), {self.workers_per_gpu} worker(s) per GPU ({self.total_workers} total), batch_size={self.batch_size}.")

        checkpoint = self.load_checkpoint()
        tasks = self.get_pending_tasks(set(checkpoint["processed"].keys()))

        if not tasks:
            print("No new files to process. Exiting.")
            return

        if self.dry_run:
            tasks = tasks[:DRY_RUN_N]
            print(f"Dry run: processing {len(tasks)} file(s).")
        else:
            print(f"Found {len(tasks)} new files. Starting processing...")

        self.process_batch(tasks, checkpoint)
        print("Processing complete.")

        if not self.no_push:
            print(f"Pushing to {self.hf_dataset}...")
            self.push_to_hub(checkpoint)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Transcribe Hebrew audio to IPA using malper/abjadsr-he.")
    parser.add_argument("--input-dir", default="./dataset_output", help="Root output dir; audio read from <input-dir>/audio/ (default: ./dataset_output)")
    parser.add_argument("--chunks-dir", default="./ipa_voxknesset", help="Directory to write VAD-split audio chunks (default: ./ipa_voxknesset)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Chunks per model.generate() call (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--workers-per-gpu", type=int, default=1, help="Worker processes per GPU (default: 1)")
    parser.add_argument("--hf-dataset", default=HF_DATASET_ID, help=f"HuggingFace dataset to push results to (default: {HF_DATASET_ID})")
    parser.add_argument("--no-push", action="store_true", help="Skip pushing results to HuggingFace")
    parser.add_argument("--dry-run", action="store_true", help=f"Process only {DRY_RUN_N} files and push")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.input_dir, "audio"), exist_ok=True)
    os.makedirs(args.chunks_dir, exist_ok=True)

    generator = Transcriber(
        input_dir=args.input_dir,
        chunks_dir=args.chunks_dir,
        workers_per_gpu=args.workers_per_gpu,
        batch_size=args.batch_size,
        hf_dataset=args.hf_dataset,
        no_push=args.no_push,
        dry_run=args.dry_run,
    )

    try:
        generator.run()
    except KeyboardInterrupt:
        print("\nProcess stopped by user.")
