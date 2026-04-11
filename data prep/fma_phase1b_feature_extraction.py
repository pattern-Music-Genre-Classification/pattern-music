"""
================================================================================
PHASE 1B: PRE-EXTRACT ALL FEATURES TO .pt FILES
================================================================================
Real-Time Music Genre Classification Pipeline — Fast Feature Cache

แก้ปัญหา: ทุก epoch ต้อง decode mp3 ใหม่ → GPU idle → training ช้า
วิธีแก้:  Extract features ครั้งเดียว → save .pt → load ตรงๆ ทุก epoch

Output structure:
  output/pt_features/
    training/   → track_{id:06d}_chunk{0-3}.pt  (4 chunks per track)
    validation/ → track_{id:06d}_chunk0.pt       (center chunk only)
    test/       → track_{id:06d}_chunk0.pt       (center chunk only)

Each .pt file contains:
  {'tabular': FloatTensor(88,), 'mel': FloatTensor(1,128,130),
   'label': int, 'track_id': int}

Author: Phase 1B — FMA Genre Classification Pipeline
================================================================================
"""

import os
import sys
import json
import pickle
import logging
import random
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# tqdm สำหรับ progress bar
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# librosa สำหรับ load mp3
try:
    import librosa
except ImportError:
    raise ImportError("librosa is required: pip install librosa")

# ตั้ง logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. CONFIGURATION & PATHS
# ==============================================================================

# Base directory: ที่ตั้งของ notebook/script นี้ (inside "data prep/")
BASE_DIR = Path(__file__).parent

# Phase 1A artifacts
PIPELINE_MODULE = BASE_DIR / "fma_phase1a_data_pipeline.py"
OUTPUT_DIR      = BASE_DIR / "output"
METADATA_CSV    = OUTPUT_DIR / "cleaned_metadata.csv"
SCALER_PKL      = OUTPUT_DIR / "tabular_scaler.pkl"
SCALING_STATS   = OUTPUT_DIR / "scaling_stats.json"
CLASS_WEIGHTS   = OUTPUT_DIR / "class_weights.pt"

# Audio directory
FMA_AUDIO_DIR = Path(r"D:\patt\project\pattern-music\FMA_Data\fma_small\fma_small")
FMA_METADATA_CSV = Path(r"D:\patt\project\pattern-music\FMA_Data\fma_metadata\fma_metadata\tracks.csv")

# Output: pre-extracted .pt features
PT_FEATURES_DIR = OUTPUT_DIR / "pt_features"

# Audio constants (เหมือน Phase 1A)
SAMPLE_RATE     = 22050
CHUNK_DURATION  = 3.0
CHUNK_SAMPLES   = int(SAMPLE_RATE * CHUNK_DURATION)  # 66,150
N_MELS          = 128
N_FFT           = 2048
HOP_LENGTH      = 512
TARGET_TIME_FRAMES = 130
MEL_SHAPE       = (1, N_MELS, TARGET_TIME_FRAMES)

# FMA genre mapping
GENRE_LABELS = [
    'Electronic', 'Experimental', 'Folk', 'Hip-Hop',
    'Instrumental', 'International', 'Pop', 'Rock'
]
NUM_CLASSES  = len(GENRE_LABELS)

# Tracks ที่ corrupt / silent ใน fma_small
CORRUPTED_TRACK_IDS = {98565, 98567, 98569, 99134, 108925, 133297, 143992}

# Training: กี่ chunks ต่อ track
N_TRAIN_CHUNKS = 4   # chunk0 = center, chunk1-3 = random positions
N_VAL_CHUNKS   = 1   # center only

# DataLoader settings
BATCH_SIZE       = 32
NUM_WORKERS      = 4
PIN_MEMORY       = True


# ==============================================================================
# 2. LOAD PHASE 1A ARTIFACTS
# ==============================================================================

def load_phase1a_artifacts():
    """
    โหลด artifacts ที่ Phase 1A สร้างไว้:
      - tabular_scaler.pkl   → sklearn StandardScaler
      - scaling_stats.json   → mel_mean, mel_std, tabular stats
      - class_weights.pt     → class weights tensor

    Returns:
        scaler: sklearn StandardScaler (fitted on training set)
        mel_mean: float — global mel mean (dB scale)
        mel_std: float  — global mel std  (dB scale)
        class_weights: torch.FloatTensor (8,)
    """
    # --- Load StandardScaler สำหรับ tabular features ---
    with open(SCALER_PKL, 'rb') as f:
        scaler = pickle.load(f)
    logger.info(f"Loaded tabular scaler: {SCALER_PKL.name}")

    # --- Load mel global statistics ---
    with open(SCALING_STATS, 'r') as f:
        stats = json.load(f)
    mel_mean = float(stats['mel_mean'])
    mel_std  = float(stats['mel_std'])
    logger.info(f"Mel global stats — mean: {mel_mean:.4f} dB, std: {mel_std:.4f} dB")

    # --- Load class weights ---
    class_weights = torch.load(CLASS_WEIGHTS, weights_only=True)
    logger.info(f"Class weights: {class_weights.numpy().round(3)}")

    return scaler, mel_mean, mel_std, class_weights


# ==============================================================================
# 3. METADATA LOADING
# ==============================================================================

def load_metadata():
    """
    โหลด metadata — ลอง cleaned_metadata.csv ก่อน, fallback ไป tracks.csv

    Returns:
        pd.DataFrame with columns: track_id, genre_top, genre_idx, split
    """
    import pandas as pd

    if METADATA_CSV.exists():
        logger.info(f"Loading cleaned metadata from: {METADATA_CSV.name}")
        df = pd.read_csv(METADATA_CSV)
        # ตรวจว่ามี columns ที่จำเป็น
        required_cols = {'track_id', 'genre_idx', 'split'}
        if required_cols.issubset(df.columns):
            logger.info(f"Loaded {len(df)} tracks from cleaned_metadata.csv")
            return df[['track_id', 'genre_idx', 'split']]

    # Fallback: load จาก tracks.csv ด้วย Phase 1A pipeline
    logger.warning("cleaned_metadata.csv not found — falling back to load_fma_tracks()")
    sys.path.insert(0, str(BASE_DIR))
    from fma_phase1a_data_pipeline import load_fma_tracks, GENRE_TO_IDX
    df = load_fma_tracks(FMA_METADATA_CSV)
    return df[['track_id', 'genre_idx', 'split']]


# ==============================================================================
# 4. DISK SPACE ESTIMATION
# ==============================================================================

def estimate_disk_usage(metadata_df) -> float:
    """
    ประมาณ disk space ที่จะใช้ก่อน extract

    Returns:
        total_gb: float — estimated total disk usage in GB
    """
    import pandas as pd

    # ขนาด raw data ต่อ .pt file
    # tabular: 88 float32 × 4 bytes = 352 bytes
    # mel:     1×128×130 float32 × 4 bytes = 66,560 bytes
    # PyTorch serialization overhead: ~3,000 bytes
    # float16 ลดขนาดเหลือ 50% เทียบกับ float32 (2 bytes แทน 4 bytes ต่อ element)
    bytes_per_file = (88 * 2) + (128 * 130 * 2) + 2000  # ≈ 35,456 bytes ≈ 35 KB (float16)

    split_counts = metadata_df['split'].value_counts()
    n_train = split_counts.get('training', 0)
    n_val   = split_counts.get('validation', 0)
    n_test  = split_counts.get('test', 0)

    # Training มี 4 chunks per track, val/test มี 1 chunk
    total_files = (n_train * N_TRAIN_CHUNKS) + (n_val * N_VAL_CHUNKS) + (n_test * N_VAL_CHUNKS)
    total_bytes = total_files * bytes_per_file
    total_gb    = total_bytes / (1024 ** 3)

    print("\n" + "=" * 60)
    print("  DISK SPACE ESTIMATION")
    print("=" * 60)
    print(f"  Per .pt file size (estimated): ~{bytes_per_file/1024:.0f} KB")
    print(f"  Training tracks: {n_train:,} × {N_TRAIN_CHUNKS} chunks = {n_train*N_TRAIN_CHUNKS:,} files")
    print(f"  Validation tracks: {n_val:,} × 1 chunk = {n_val:,} files")
    print(f"  Test tracks: {n_test:,} × 1 chunk = {n_test:,} files")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Total files: {total_files:,}")
    print(f"  Estimated disk usage: {total_gb:.2f} GB")
    print("=" * 60)

    return total_gb


# ==============================================================================
# 5. CORE EXTRACTION FUNCTIONS
# ==============================================================================

def get_audio_path(audio_dir: Path, track_id: int) -> Path:
    """สร้าง path ไปยัง mp3 file ตาม FMA directory structure"""
    tid_str = f"{track_id:06d}"
    return audio_dir / tid_str[:3] / f"{tid_str}.mp3"


def extract_chunk(audio: np.ndarray, chunk_samples: int,
                  mode: str = 'center',
                  rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    ตัด chunk ขนาด chunk_samples จาก audio

    Args:
        audio:        1D numpy array — full audio
        chunk_samples: จำนวน samples ที่ต้องการ (66,150)
        mode:         'center' → ตัดกลาง (deterministic)
                      'random' → ตัดสุ่ม (data augmentation)
        rng:          numpy RandomState สำหรับ reproducible random

    Returns:
        1D numpy array shape (chunk_samples,)
    """
    total = len(audio)

    # เพลงสั้นกว่า 3 วินาที → zero-pad ทางขวา
    if total < chunk_samples:
        padded = np.zeros(chunk_samples, dtype=np.float32)
        padded[:total] = audio
        return padded

    if mode == 'center':
        start = (total - chunk_samples) // 2
    else:  # random
        if rng is None:
            rng = np.random.RandomState()
        max_start = total - chunk_samples
        start = rng.randint(0, max_start + 1)

    return audio[start:start + chunk_samples].astype(np.float32)


def apply_mel_global_norm(mel_arr: np.ndarray, mel_mean: float, mel_std: float) -> np.ndarray:
    """
    Apply global normalization บน mel spectrogram (หลัง power_to_db)
    mel_arr shape: (1, 128, 130)

    สูตร: (mel - global_mean) / global_std
    """
    return ((mel_arr - mel_mean) / mel_std).astype(np.float32)


def apply_tabular_scaling(tabular_arr: np.ndarray, scaler) -> np.ndarray:
    """
    Apply StandardScaler transform บน tabular feature vector
    tabular_arr shape: (88,)
    """
    return scaler.transform(tabular_arr.reshape(1, -1)).flatten().astype(np.float32)


def extract_and_save_track(
    track_id: int,
    audio_dir: Path,
    genre_idx: int,
    split: str,
    out_dir: Path,
    scaler,
    mel_mean: float,
    mel_std: float,
) -> Tuple[int, List[str], Optional[str]]:
    """
    Load mp3 ครั้งเดียว → extract chunks → scale → save .pt files

    Training: 4 chunks (chunk0=center, chunk1-3=random)
    Val/Test: 1 chunk  (chunk0=center)

    Returns:
        (track_id, saved_paths, error_msg or None)
    """
    # Import ฟังก์ชัน extraction จาก Phase 1A
    sys.path.insert(0, str(BASE_DIR))
    from fma_phase1a_data_pipeline import extract_tabular_features, extract_mel_spectrogram

    audio_path = get_audio_path(audio_dir, track_id)

    # ตรวจสอบ file ก่อน load
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        return track_id, [], f"File not found or empty: {audio_path.name}"

    try:
        # Load mp3 ครั้งเดียว (librosa) — ส่วนที่ช้าที่สุด
        audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        return track_id, [], f"librosa.load failed: {e}"

    # กำหนดว่าจะ extract กี่ chunks และอย่างไร
    is_training = (split == 'training')
    n_chunks    = N_TRAIN_CHUNKS if is_training else N_VAL_CHUNKS

    saved_paths = []
    rng = np.random.RandomState(seed=track_id)  # reproducible per track

    try:
        for chunk_idx in range(n_chunks):
            # chunk0 เสมอ = center; chunk1+ = random (training only)
            mode = 'center' if chunk_idx == 0 else 'random'
            chunk = extract_chunk(audio, CHUNK_SAMPLES, mode=mode, rng=rng)

            # Extract features (raw, unscaled)
            tabular_raw = extract_tabular_features(chunk, SAMPLE_RATE)   # (88,)
            mel_raw     = extract_mel_spectrogram(chunk, SAMPLE_RATE)    # (1,128,130)

            # Apply scaling
            tabular_scaled = apply_tabular_scaling(tabular_raw, scaler)
            mel_scaled     = apply_mel_global_norm(mel_raw, mel_mean, mel_std)

            # ตรวจสอบ NaN/Inf ก่อน save
            if np.any(np.isnan(tabular_scaled)) or np.any(np.isinf(tabular_scaled)):
                return track_id, saved_paths, f"NaN/Inf in tabular at chunk {chunk_idx}"
            if np.any(np.isnan(mel_scaled)) or np.any(np.isinf(mel_scaled)):
                return track_id, saved_paths, f"NaN/Inf in mel at chunk {chunk_idx}"

            # สร้าง dict สำหรับ save — บันทึกเป็น float16 เพื่อลด disk ~50%
            # (float32 → float16: 4 bytes → 2 bytes ต่อ element)
            data = {
                'tabular':  torch.from_numpy(tabular_scaled).to(torch.float16),  # (88,)
                'mel':      torch.from_numpy(mel_scaled).to(torch.float16),       # (1,128,130)
                'label':    genre_idx,                                             # int
                'track_id': track_id,                                              # int
            }

            # Save .pt file
            filename = f"track_{track_id:06d}_chunk{chunk_idx}.pt"
            save_path = out_dir / filename
            torch.save(data, save_path)
            saved_paths.append(str(save_path))

    except Exception as e:
        return track_id, saved_paths, f"Extraction error: {e}"

    return track_id, saved_paths, None  # None = success


# ==============================================================================
# 6. RUN PRE-EXTRACTION
# ==============================================================================

def run_extraction(
    metadata_df,
    audio_dir: Path = FMA_AUDIO_DIR,
    scaler=None,
    mel_mean: float = -41.75,
    mel_std: float = 13.74,
    max_workers: int = 4,
    overwrite: bool = False,
) -> Dict[str, Dict]:
    """
    Extract features สำหรับทุก tracks แล้ว save เป็น .pt files

    Args:
        metadata_df: DataFrame with track_id, genre_idx, split
        audio_dir:   path ไปยัง fma_small audio directory
        scaler:      sklearn StandardScaler (fitted)
        mel_mean:    global mel mean (dB)
        mel_std:     global mel std (dB)
        max_workers: number of ThreadPoolExecutor workers
        overwrite:   ถ้า True จะ overwrite .pt files ที่มีอยู่แล้ว

    Returns:
        results dict: {split: {'success': int, 'failed': int, 'errors': list}}
    """
    if scaler is None:
        scaler, mel_mean, mel_std, _ = load_phase1a_artifacts()

    # สร้าง output directories
    for split in ['training', 'validation', 'test']:
        (PT_FEATURES_DIR / split).mkdir(parents=True, exist_ok=True)

    results = {
        split: {'success': 0, 'failed': 0, 'errors': []}
        for split in ['training', 'validation', 'test']
    }

    print("\n" + "=" * 60)
    print("  PHASE 1B: FEATURE EXTRACTION")
    print("=" * 60)

    # Group metadata by split เพื่อ process แยก (พร้อม progress bar)
    for split in ['training', 'validation', 'test']:
        split_df   = metadata_df[metadata_df['split'] == split].reset_index(drop=True)
        out_dir    = PT_FEATURES_DIR / split
        n_tracks   = len(split_df)
        n_chunks   = N_TRAIN_CHUNKS if split == 'training' else N_VAL_CHUNKS
        n_expected = n_tracks * n_chunks

        print(f"\n[{split.upper()}] {n_tracks} tracks → {n_expected} .pt files expected")

        # Skip tracks ที่ extract ไปแล้ว (ถ้า overwrite=False)
        rows_to_process = []
        skipped = 0
        for _, row in split_df.iterrows():
            track_id = row['track_id']
            if track_id in CORRUPTED_TRACK_IDS:
                skipped += 1
                continue
            # ตรวจว่า chunk0 มีอยู่แล้วหรือเปล่า
            chunk0_path = out_dir / f"track_{track_id:06d}_chunk0.pt"
            if not overwrite and chunk0_path.exists():
                skipped += 1
                results[split]['success'] += n_chunks  # นับว่า success แล้ว
                continue
            rows_to_process.append(row)

        if skipped > 0:
            logger.info(f"  Skipped {skipped} tracks (already extracted or corrupted)")

        if not rows_to_process:
            print(f"  All {n_tracks} tracks already extracted — skipping")
            continue

        # Extract ด้วย ProcessPoolExecutor — librosa FFT/mel เป็น CPU-bound
        # ThreadPoolExecutor จะโดน GIL block → ไม่ได้ parallel จริงๆ
        # ProcessPoolExecutor ทำ true multiprocessing → ใช้ CPU cores ได้เต็มที่
        # ⚠️ Windows: function ที่ submit ต้องอยู่ใน importable module (ไม่ใช่ cell/lambda)
        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for row in rows_to_process:
                future = executor.submit(
                    extract_and_save_track,
                    int(row['track_id']),
                    audio_dir,
                    int(row['genre_idx']),
                    split,
                    out_dir,
                    scaler,
                    mel_mean,
                    mel_std,
                )
                futures[future] = int(row['track_id'])

            # tqdm progress bar — update per track
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"  {split[:5]}",
                unit="track",
                ncols=80,
            )
            for future in pbar:
                track_id, saved_paths, error = future.result()
                if error is None:
                    results[split]['success'] += len(saved_paths)
                else:
                    results[split]['failed'] += 1
                    results[split]['errors'].append(f"track_{track_id:06d}: {error}")
                    logger.debug(f"  Error — track {track_id}: {error}")

        # สรุปผล
        s = results[split]
        print(f"  ✓ Success: {s['success']} files saved")
        if s['failed'] > 0:
            print(f"  ✗ Failed:  {s['failed']} tracks")
            for err in s['errors'][:5]:  # แสดงแค่ 5 errors แรก
                print(f"    - {err}")
            if len(s['errors']) > 5:
                print(f"    ... and {len(s['errors'])-5} more")

    print("\n" + "=" * 60)
    print("  EXTRACTION COMPLETE")
    print("=" * 60)
    total_success = sum(r['success'] for r in results.values())
    total_failed  = sum(r['failed']  for r in results.values())
    print(f"  Total .pt files saved: {total_success:,}")
    print(f"  Total tracks failed:   {total_failed:,}")

    # คำนวณ actual disk usage
    actual_bytes = sum(
        f.stat().st_size
        for split in ['training', 'validation', 'test']
        for f in (PT_FEATURES_DIR / split).glob("*.pt")
    )
    print(f"  Actual disk usage:     {actual_bytes / (1024**3):.3f} GB")

    return results


# ==============================================================================
# 7. TENSOR-BASED AUGMENTATION (ไม่ใช้ librosa — เร็วกว่า)
# ==============================================================================

def spec_augment(
    mel: torch.Tensor,
    freq_mask_param: int = 15,
    time_mask_param: int = 20,
    n_freq_masks: int = 2,
    n_time_masks: int = 2,
) -> torch.Tensor:
    """
    SpecAugment (Park et al., 2019) — mask random bands บน mel spectrogram tensor

    Args:
        mel: FloatTensor shape (1, n_mels, n_frames) = (1, 128, 130)
        freq_mask_param: max width ของ frequency mask
        time_mask_param: max width ของ time mask
        n_freq_masks:    จำนวน frequency masks
        n_time_masks:    จำนวน time masks

    Returns:
        FloatTensor (1, 128, 130) — same shape, with masked regions
    """
    mel = mel.clone()
    # fill ด้วย 0.0 — data ถูก normalize แล้ว (mean≈0, std≈1)
    # 0.0 = mean ของ normalized data → masked regions ไม่ distort distribution
    fill_val = 0.0
    _, n_mels, n_frames = mel.shape

    # --- Frequency Masking (แนวนอน — mask freq bins) ---
    for _ in range(n_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(0, n_mels - f))
        mel[:, f0:f0 + f, :] = fill_val

    # --- Time Masking (แนวตั้ง — mask time frames) ---
    for _ in range(n_time_masks):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, max(0, n_frames - t))
        mel[:, :, t0:t0 + t] = fill_val

    return mel


def add_gaussian_noise(mel: torch.Tensor, noise_std: Optional[float] = None) -> torch.Tensor:
    """
    เพิ่ม Gaussian noise บน mel spectrogram tensor (normalized data)

    เนื่องจาก data ถูก normalize แล้ว (mean≈0, std≈1) ไม่จำเป็นต้องคำนวณ SNR —
    แค่ sample noise_std จาก [0.05, 0.15] ซึ่ง correspond กับ ~7-20% ของ signal std

    Args:
        mel:       FloatTensor shape (1, 128, 130)
        noise_std: std ของ Gaussian noise — ถ้า None จะสุ่มใน [0.05, 0.15]

    Returns:
        FloatTensor (1, 128, 130) — mel + noise
    """
    if noise_std is None:
        noise_std = random.uniform(0.05, 0.15)

    noise = torch.randn_like(mel) * noise_std
    return mel + noise


# ==============================================================================
# 8. PreExtractedFMADataset
# ==============================================================================

class PreExtractedFMADataset(Dataset):
    """
    Fast PyTorch Dataset ที่ load .pt files แทน mp3

    __getitem__ แค่:
      1. torch.load() → dict
      2. apply augmentation (training only)
      3. return (tabular, mel, label)

    เร็วกว่า FMADataset เดิม 10-50x เพราะไม่ต้อง decode mp3

    Args:
        pt_dir:  path ไปยัง directory ของ split (เช่น output/pt_features/training/)
        augment: ถ้า True จะ apply SpecAugment + Gaussian noise
    """

    def __init__(self, pt_dir: Path, augment: bool = False):
        self.pt_dir  = Path(pt_dir)
        self.augment = augment

        # Glob ทุก .pt files ใน directory
        self.files = sorted(self.pt_dir.glob("*.pt"))

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No .pt files found in {self.pt_dir}\n"
                f"Run run_extraction() first to generate pre-extracted features."
            )

        logger.info(f"PreExtractedFMADataset: {len(self.files)} files from {self.pt_dir.name}/")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tabular:  FloatTensor (88,)
            mel:      FloatTensor (1, 128, 130)
            label:    LongTensor  scalar
        """
        # Load .pt file (fast — no mp3 decode)
        data     = torch.load(self.files[idx], weights_only=True)
        # .to(float32): convert float16 → float32 ก่อนส่งเข้า model
        tabular  = data['tabular'].to(torch.float32)   # (88,)
        mel      = data['mel'].to(torch.float32)       # (1, 128, 130)
        label    = torch.tensor(data['label'], dtype=torch.long)

        # Augmentation สำหรับ training only (บน tensor — ไม่ต้องใช้ librosa)
        if self.augment:
            # SpecAugment: frequency + time masking — p=0.5
            if random.random() < 0.5:
                mel = spec_augment(mel)
            # Gaussian noise บน mel — p=0.3
            if random.random() < 0.3:
                mel = add_gaussian_noise(mel)

        return tabular, mel, label


# ==============================================================================
# 9. FAST DATALOADER FACTORY
# ==============================================================================

def create_fast_dataloaders(
    pt_base_dir: Path = PT_FEATURES_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    use_weighted_sampler: bool = True,
    class_weights: Optional[torch.Tensor] = None,
    metadata_df=None,   # ถ้าส่งมา จะใช้ filename parsing แทน torch.load ต่อไฟล์
) -> Dict[str, DataLoader]:
    """
    สร้าง DataLoaders สำหรับทุก splits จาก pre-extracted .pt files

    Args:
        pt_base_dir:          path ไปยัง output/pt_features/
        batch_size:           batch size
        num_workers:          parallel workers สำหรับ DataLoader
        use_weighted_sampler: ใช้ WeightedRandomSampler สำหรับ training
        class_weights:        FloatTensor (8,) — ถ้า None จะ load จาก class_weights.pt

    Returns:
        dict: {'training': DataLoader, 'validation': DataLoader, 'test': DataLoader}
    """
    pt_base_dir = Path(pt_base_dir)

    # Load class weights ถ้าไม่ได้ส่งมา
    if class_weights is None and CLASS_WEIGHTS.exists():
        class_weights = torch.load(CLASS_WEIGHTS, weights_only=True)

    loaders = {}

    for split in ['training', 'validation', 'test']:
        split_dir = pt_base_dir / split
        is_train  = (split == 'training')

        dataset = PreExtractedFMADataset(
            pt_dir  = split_dir,
            augment = is_train,
        )

        # WeightedRandomSampler สำหรับ training (optional — dataset balanced อยู่แล้ว)
        sampler = None
        if is_train and use_weighted_sampler and class_weights is not None:
            if metadata_df is not None:
                # Fast path: parse track_id จาก filename → O(1) dict lookup
                # ไม่ต้อง torch.load() ทุกไฟล์ — หลีกเลี่ยง O(N) disk reads
                tid_to_label = dict(zip(
                    metadata_df['track_id'].astype(int),
                    metadata_df['genre_idx'].astype(int),
                ))
                sample_weights = [
                    class_weights[tid_to_label.get(int(f.stem.split('_')[1]), 0)].item()
                    for f in dataset.files
                ]
            else:
                # Slow fallback: load ทุกไฟล์เพื่ออ่าน label
                # ⚠️ หลีกเลี่ยงถ้าทำได้ — O(N) disk reads ทำให้ช้ามาก (~หลายชั่วโมง!)
                logger.warning(
                    "WeightedRandomSampler: reading every .pt file — "
                    "pass metadata_df to create_fast_dataloaders() for fast path"
                )
                sample_weights = [
                    class_weights[torch.load(f, weights_only=True)['label']].item()
                    for f in dataset.files
                ]

            sampler = WeightedRandomSampler(
                weights     = torch.tensor(sample_weights, dtype=torch.double),
                num_samples = len(dataset),
                replacement = True,
            )

        loader = DataLoader(
            dataset,
            batch_size         = batch_size,
            sampler            = sampler,
            shuffle            = (is_train and sampler is None),  # shuffle ถ้าไม่มี sampler
            num_workers        = num_workers,
            pin_memory         = PIN_MEMORY,
            drop_last          = is_train,           # drop incomplete batch ตอน train
            # persistent_workers=True สำหรับ .py script (ลด worker spawn overhead)
            # ⚠️ ใน Jupyter/notebook บน Windows ให้ใช้ False — zombie processes / RAM leak
            persistent_workers = (num_workers > 0),  # True เมื่อรันเป็น script
        )

        loaders[split] = loader
        logger.info(
            f"[{split}] DataLoader ready: {len(dataset)} samples, "
            f"{len(loader)} batches, augment={is_train}"
        )

    return loaders


# ==============================================================================
# 10. VERIFICATION
# ==============================================================================

def verify_fast_dataloaders(loaders: Dict[str, DataLoader]) -> None:
    """
    ตรวจสอบ shapes, NaN/Inf, และ benchmark speedup vs mp3 loading

    Args:
        loaders: dict จาก create_fast_dataloaders()
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("  VERIFICATION: PreExtractedFMADataset")
    print("=" * 60)

    # --- Shape verification ---
    for split in ['training', 'validation']:
        loader = loaders[split]
        tabular_batch, mel_batch, label_batch = next(iter(loader))

        actual_bs = tabular_batch.shape[0]
        print(f"\n[{split.upper()}] Batch shapes:")
        print(f"  tabular: {tuple(tabular_batch.shape)}  (expected: ({actual_bs}, 88))")
        print(f"  mel:     {tuple(mel_batch.shape)}  (expected: ({actual_bs}, 1, 128, 130))")
        print(f"  label:   {tuple(label_batch.shape)}  (expected: ({actual_bs},))")

        # Assertions
        assert tabular_batch.shape == (actual_bs, 88), \
            f"Tabular shape mismatch: {tabular_batch.shape}"
        assert mel_batch.shape == (actual_bs, 1, 128, 130), \
            f"Mel shape mismatch: {mel_batch.shape}"
        assert label_batch.shape == (actual_bs,), \
            f"Label shape mismatch: {label_batch.shape}"

        # NaN/Inf check
        assert not torch.isnan(tabular_batch).any(), "NaN in tabular!"
        assert not torch.isnan(mel_batch).any(),     "NaN in mel!"
        assert not torch.isinf(tabular_batch).any(), "Inf in tabular!"
        assert not torch.isinf(mel_batch).any(),     "Inf in mel!"

        print(f"  ✓ All shape assertions PASSED, no NaN/Inf")

    # --- Speedup benchmark: .pt vs mp3 ---
    print("\n[SPEEDUP BENCHMARK]")
    import pandas as pd
    from fma_phase1a_data_pipeline import FMADataset

    try:
        metadata_df = load_metadata()
        train_pt_loader = loaders['training']

        # เวลา .pt loading (5 batches)
        n_bench = 5
        t0 = time.perf_counter()
        for i, _ in enumerate(train_pt_loader):
            if i >= n_bench - 1:
                break
        pt_time = time.perf_counter() - t0

        # เวลา mp3 loading (5 batches — num_workers=0 เพื่อความยุติธรรม)
        mp3_dataset = FMADataset(metadata_df, FMA_AUDIO_DIR, split='training')
        mp3_loader  = DataLoader(mp3_dataset, batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=0, pin_memory=False)

        t0 = time.perf_counter()
        for i, _ in enumerate(mp3_loader):
            if i >= n_bench - 1:
                break
        mp3_time = time.perf_counter() - t0

        speedup = mp3_time / pt_time if pt_time > 0 else float('inf')
        print(f"  .pt  loading ({n_bench} batches): {pt_time:.2f}s")
        print(f"  mp3  loading ({n_bench} batches): {mp3_time:.2f}s")
        print(f"  ✓ Speedup: {speedup:.1f}x {'(≥10x target MET!)' if speedup >= 10 else '(< 10x target)'}")

    except Exception as e:
        print(f"  Speedup benchmark skipped: {e}")

    # --- Plot 4 sample spectrograms ---
    print("\n[PLOTTING] 4 sample spectrograms from .pt dataset...")
    train_loader = loaders['training']
    tabular_b, mel_b, label_b = next(iter(train_loader))

    genre_labels = GENRE_LABELS

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(4):
        spec     = mel_b[i, 0].numpy()   # (128, 130)
        genre    = genre_labels[label_b[i].item()]
        im = axes[i].imshow(
            spec, aspect='auto', origin='lower',
            cmap='magma',
            extent=[0, CHUNK_DURATION, 0, SAMPLE_RATE // 2],
        )
        axes[i].set_title(f'Genre: {genre}', fontsize=11)
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Frequency (Hz)')
        fig.colorbar(im, ax=axes[i], format='%.1f')

    fig.suptitle('Sample Mel Spectrograms — Pre-Extracted .pt Dataset (Scaled)', fontsize=13)
    plt.tight_layout()

    save_path = OUTPUT_DIR / "phase1b_sample_spectrograms.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Spectrograms saved to: {save_path.name}")

    print("\n" + "=" * 60)
    print("  VERIFICATION COMPLETE — PreExtractedFMADataset is READY")
    print("=" * 60)


# ==============================================================================
# MAIN (เมื่อรันตรงๆ)
# ==============================================================================

def main():
    """
    Entry point เมื่อรัน script โดยตรง:
      python fma_phase1b_feature_extraction.py

    ขั้นตอน:
      1. Load Phase 1A artifacts
      2. Load metadata
      3. แสดง disk estimation แล้วถาม confirm
      4. Extract features → save .pt files
      5. Create DataLoaders + verify
    """
    # 1. Load artifacts
    scaler, mel_mean, mel_std, class_weights = load_phase1a_artifacts()

    # 2. Load metadata
    import pandas as pd
    metadata_df = load_metadata()
    print(f"\nMetadata loaded: {len(metadata_df)} tracks")
    print(metadata_df['split'].value_counts().to_string())

    # 3. Disk space estimation + confirmation
    total_gb = estimate_disk_usage(metadata_df)
    confirm = input(f"\nProceed with extraction (~{total_gb:.2f} GB)? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Extraction cancelled.")
        return

    # 4. Extract features
    results = run_extraction(
        metadata_df = metadata_df,
        audio_dir   = FMA_AUDIO_DIR,
        scaler      = scaler,
        mel_mean    = mel_mean,
        mel_std     = mel_std,
        max_workers = 4,
        overwrite   = False,
    )

    # 5. Create DataLoaders + Verify
    loaders = create_fast_dataloaders(
        pt_base_dir         = PT_FEATURES_DIR,
        batch_size          = BATCH_SIZE,
        num_workers         = NUM_WORKERS,
        use_weighted_sampler = True,
        class_weights       = class_weights,
    )
    verify_fast_dataloaders(loaders)


if __name__ == "__main__":
    main()
