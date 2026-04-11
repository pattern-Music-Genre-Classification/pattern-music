"""
================================================================================
PHASE 1A: FMA NATIVE-SHAPE DATA PIPELINE
================================================================================
Real-Time Music Genre Classification Pipeline — Data Foundation
Target: NVIDIA GPU (RTX 4060) | FMA Small Dataset | 8 Genres

This script handles:
  1. Global configuration & path setup (Windows-safe via pathlib)
  2. Metadata parsing with FMA's tricky multi-level CSV headers
  3. Advanced feature extraction (tabular + mel spectrogram)
  4. PyTorch Dataset & DataLoader with chunking strategy
  5. Verification block with shape checks & spectrogram visualization

NO model training code — this is purely the data pipeline.

Author: Senior Audio ML Data Engineer
================================================================================
"""

import os
import warnings
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend สำหรับ save plot โดยไม่ต้องเปิดหน้าต่าง
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ตั้ง logging เพื่อ track ปัญหาระหว่าง data loading
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. GLOBAL CONFIGURATION
# ==============================================================================

# --- Paths (Windows-safe via pathlib) ---
# เปลี่ยน path ตรงนี้ให้ตรงกับเครื่องของคุณ
FMA_AUDIO_DIR = Path(r"D:\patt\project\pattern-music\FMA_Data\fma_small\fma_small")
FMA_METADATA_CSV = Path(r"D:\patt\project\pattern-music\FMA_Data\fma_metadata\fma_metadata\tracks.csv")

# --- Audio Hyperparameters ---
SAMPLE_RATE = 22050          # FMA native sample rate
CHUNK_DURATION = 3.0         # วินาที — แต่ละ chunk ที่ตัดจากเพลง
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)  # = 66,150 samples

# --- Mel Spectrogram Hyperparameters ---
N_MELS = 128                 # จำนวน mel frequency bins
N_FFT = 2048                 # FFT window size
HOP_LENGTH = 512             # hop length สำหรับ STFT
# CRITICAL SHAPE: ด้วย sr=22050, hop=512, chunk=3s → time frames ≈ 130
# เราจะ enforce ให้ได้ exactly 130 เสมอ (pad/truncate)
TARGET_TIME_FRAMES = 130
MEL_SHAPE = (1, N_MELS, TARGET_TIME_FRAMES)  # (C, H, W) = (1, 128, 130)

# --- DataLoader Hyperparameters ---
BATCH_SIZE = 32
NUM_WORKERS = 4              # parallel data loading — ปรับตาม CPU cores
PIN_MEMORY = True            # เร่ง GPU transfer

# --- FMA Corrupted/Missing Tracks ---
# tracks เหล่านี้เป็น 0 bytes หรือ corrupted ใน fma_small
# hardcode ไว้เลยเพื่อไม่ต้อง try/except ตอน runtime
CORRUPTED_TRACK_IDS = {98565, 98567, 98569, 99134, 108925, 133297, 143992}

# --- Genre Mapping ---
# FMA small มี 8 genres — map เป็น integer labels
GENRE_LABELS = [
    'Electronic', 'Experimental', 'Folk', 'Hip-Hop',
    'Instrumental', 'International', 'Pop', 'Rock'
]
GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(GENRE_LABELS)}
NUM_CLASSES = len(GENRE_LABELS)

# --- Output ---
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ==============================================================================
# 2. METADATA PARSING
# ==============================================================================

def load_fma_tracks(csv_path: Path) -> pd.DataFrame:
    """
    โหลด tracks.csv ของ FMA ซึ่งมี multi-level column headers (2 rows)
    แล้ว filter เฉพาะ fma_small subset พร้อม clean data

    FMA tracks.csv structure:
      - Row 0-1: Multi-level headers (e.g., ('track', 'genre_top'), ('set', 'subset'))
      - Index: track_id (integer)

    Returns:
        pd.DataFrame พร้อม columns: track_id, genre_top, genre_idx, split
    """
    logger.info(f"Loading metadata from: {csv_path}")

    # --- โหลด CSV แบบ multi-level header (เหมือน FMA utils.load) ---
    # header=[0, 1] อ่าน 2 แถวแรกเป็น column names → MultiIndex columns
    tracks = pd.read_csv(csv_path, index_col=0, header=[0, 1])

    # ดึง columns ที่เราต้องการ
    # ('track', 'genre_top') = genre label
    # ('set', 'subset')      = small/medium/large
    # ('set', 'split')        = training/validation/test
    genre_col = ('track', 'genre_top')
    subset_col = ('set', 'subset')
    split_col = ('set', 'split')

    # --- Filter เฉพาะ fma_small ---
    small_mask = tracks[subset_col] == 'small'
    df = tracks.loc[small_mask, [genre_col, split_col]].copy()
    df.columns = ['genre_top', 'split']  # flatten column names
    df.index.name = 'track_id'

    logger.info(f"FMA small tracks (before cleaning): {len(df)}")

    # --- Clean: drop rows ที่ genre เป็น NaN ---
    df = df.dropna(subset=['genre_top'])

    # --- Clean: drop corrupted tracks ---
    corrupted_in_dataset = CORRUPTED_TRACK_IDS.intersection(set(df.index))
    if corrupted_in_dataset:
        logger.warning(f"Dropping {len(corrupted_in_dataset)} corrupted tracks: {corrupted_in_dataset}")
        df = df.drop(index=list(corrupted_in_dataset), errors='ignore')

    # --- Map genre strings to integer labels ---
    # ตรวจว่า genres ใน dataset ตรงกับ GENRE_LABELS ที่เรา define
    dataset_genres = sorted(df['genre_top'].unique())
    assert dataset_genres == sorted(GENRE_LABELS), \
        f"Genre mismatch! Dataset: {dataset_genres}, Expected: {sorted(GENRE_LABELS)}"

    df['genre_idx'] = df['genre_top'].map(GENRE_TO_IDX)

    # --- สรุปข้อมูล ---
    logger.info(f"FMA small tracks (after cleaning): {len(df)}")
    logger.info(f"Splits: {df['split'].value_counts().to_dict()}")
    logger.info(f"Genres: {df['genre_top'].value_counts().to_dict()}")

    return df.reset_index()  # track_id กลับมาเป็น column


def get_audio_path(audio_dir: Path, track_id: int) -> Path:
    """
    สร้าง path ไปยัง mp3 file ตาม FMA directory structure:
      audio_dir / <first 3 digits as folder> / <track_id zero-padded to 6>.mp3

    Example: track_id=2 → audio_dir/000/000002.mp3
             track_id=12345 → audio_dir/012/012345.mp3
    """
    tid_str = f"{track_id:06d}"
    return audio_dir / tid_str[:3] / f"{tid_str}.mp3"


# ==============================================================================
# 3. ADVANCED FEATURE EXTRACTION (librosa)
# ==============================================================================

def extract_tabular_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    คำนวณ dense feature vector จาก 1D audio array (exactly 3 seconds)
    สำหรับ traditional ML models (Random Forest baseline)

    Features extracted (mean + std across time axis):
      - MFCCs:              20 coefficients  × 2 (mean, std) = 40
      - Spectral Centroid:  1 × 2 = 2
      - Spectral Bandwidth: 1 × 2 = 2
      - Spectral Rolloff:   1 × 2 = 2
      - Chroma STFT:        12 × 2 = 24
      - Zero Crossing Rate: 1 × 2 = 2
      - RMS Energy:         1 × 2 = 2
      - Spectral Contrast:  7 × 2 = 14
      ─────────────────────────────────────
      Total:                88 features

    Args:
        audio: 1D numpy array, shape (CHUNK_SAMPLES,)
        sr: sample rate

    Returns:
        1D numpy array, shape (88,) — all features concatenated
    """
    features = []

    # --- MFCCs (20 coefficients) ---
    # MFCCs จับ timbral texture ของเสียง — สำคัญที่สุดสำหรับ genre classification
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(mfccs.mean(axis=1))  # (20,)
    features.append(mfccs.std(axis=1))   # (20,)

    # --- Spectral Centroid ---
    # "center of mass" ของ spectrum — เพลง bright (pop) จะสูง, เพลง dark (rock) จะต่ำ
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(centroid.mean(axis=1))  # (1,)
    features.append(centroid.std(axis=1))   # (1,)

    # --- Spectral Bandwidth ---
    # ความกว้างของ spectrum — บอก "richness" ของเสียง
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(bandwidth.mean(axis=1))  # (1,)
    features.append(bandwidth.std(axis=1))   # (1,)

    # --- Spectral Rolloff ---
    # frequency ที่ต่ำกว่านี้มี 85% ของ spectral energy — แยก voiced/unvoiced ได้
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(rolloff.mean(axis=1))  # (1,)
    features.append(rolloff.std(axis=1))   # (1,)

    # --- Chroma STFT (12 pitch classes) ---
    # จับ harmonic content — แยก genre ที่มี chord progression ต่างกัน
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(chroma.mean(axis=1))  # (12,)
    features.append(chroma.std(axis=1))   # (12,)

    # --- Zero Crossing Rate ---
    # จำนวนครั้งที่ signal ข้าม 0 — สูงสำหรับ percussive / noisy audio
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=HOP_LENGTH)
    features.append(zcr.mean(axis=1))  # (1,)
    features.append(zcr.std(axis=1))   # (1,)

    # --- RMS Energy ---
    # ความดังโดยรวม — แยก genre ที่ dynamic range ต่างกัน
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
    features.append(rms.mean(axis=1))  # (1,)
    features.append(rms.std(axis=1))   # (1,)

    # --- Spectral Contrast (7 bands) ---
    # ความต่างระหว่าง peak และ valley ในแต่ละ frequency band
    # ช่วยแยกเพลงที่ "flat" (electronic) กับ "dynamic" (classical)
    contrast = librosa.feature.spectral_contrast(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_bands=6
    )  # returns (7, T) — 6 bands + 1 valley
    features.append(contrast.mean(axis=1))  # (7,)
    features.append(contrast.std(axis=1))   # (7,)

    # --- Concatenate ทุก features เป็น 1D vector ---
    feature_vector = np.concatenate(features, axis=0).astype(np.float32)

    return feature_vector  # shape: (88,)


def extract_mel_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    สร้าง log-scaled Mel Spectrogram จาก 1D audio array
    พร้อม enforce exact shape (1, 128, 130)

    Pipeline:
      audio → STFT → mel filterbank → power to dB → pad/truncate → add channel dim

    CRITICAL: output shape ต้องเป็น (1, 128, 130) เสมอ
      - 1   = channel dimension (grayscale)
      - 128 = mel frequency bins
      - 130 = time frames (จาก 3s audio, sr=22050, hop=512)

    Args:
        audio: 1D numpy array, shape (CHUNK_SAMPLES,)
        sr: sample rate

    Returns:
        numpy array, shape (1, 128, 130)
    """
    # --- Mel spectrogram (power) ---
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=sr // 2  # Nyquist frequency
    )

    # --- Convert power to dB scale ---
    # ทำให้ค่าอยู่ใน range ที่ neural network จัดการง่าย
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # --- Strict shape enforcement: time axis must be exactly 130 ---
    # floating point ของ audio length อาจทำให้ได้ 129 หรือ 131 frames
    n_frames = log_mel.shape[1]

    if n_frames > TARGET_TIME_FRAMES:
        # Truncate: ตัดส่วนเกินทิ้ง
        log_mel = log_mel[:, :TARGET_TIME_FRAMES]
    elif n_frames < TARGET_TIME_FRAMES:
        # Pad: เติม 0 (silence) ทางขวา
        pad_width = TARGET_TIME_FRAMES - n_frames
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant', constant_values=-80.0)
        # -80 dB ≈ silence ใน log scale

    # --- Add channel dimension: (128, 130) → (1, 128, 130) ---
    log_mel = log_mel[np.newaxis, :, :]

    return log_mel.astype(np.float32)


# ==============================================================================
# 4. FMA PYTORCH DATASET
# ==============================================================================

class FMADataset(Dataset):
    """
    PyTorch Dataset สำหรับ FMA Small

    แต่ละ item return tuple ของ 3 tensors:
      (tabular_features, mel_spectrogram, label)
      (88,)              (1, 128, 130)     scalar

    Chunking Strategy:
      - Training:   สุ่ม random 3s slice (temporal data augmentation)
      - Val/Test:   ตัด center 3s slice (deterministic — ผลซ้ำได้)

    ถ้าเพลงสั้นกว่า 3 วินาที → pad ด้วย zeros
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        audio_dir: Path,
        split: str = 'training',
        sample_rate: int = SAMPLE_RATE,
        chunk_samples: int = CHUNK_SAMPLES,
    ):
        """
        Args:
            metadata_df: DataFrame จาก load_fma_tracks() — มี track_id, genre_top, genre_idx, split
            audio_dir: path ไปยัง FMA audio directory
            split: 'training', 'validation', หรือ 'test'
            sample_rate: target sample rate
            chunk_samples: จำนวน samples ใน 1 chunk (3s = 66150)
        """
        self.audio_dir = audio_dir
        self.sr = sample_rate
        self.chunk_samples = chunk_samples
        self.split = split
        self.is_training = (split == 'training')

        # --- Filter สำหรับ split ที่ต้องการ ---
        self.df = metadata_df[metadata_df['split'] == split].reset_index(drop=True)

        # --- Verify ว่า audio files มีอยู่จริง ---
        valid_indices = []
        missing_count = 0
        for idx, row in self.df.iterrows():
            audio_path = get_audio_path(self.audio_dir, row['track_id'])
            if audio_path.exists() and audio_path.stat().st_size > 0:
                valid_indices.append(idx)
            else:
                missing_count += 1

        if missing_count > 0:
            logger.warning(f"[{split}] {missing_count} audio files missing/empty — skipped")

        self.df = self.df.loc[valid_indices].reset_index(drop=True)
        logger.info(f"[{split}] Dataset ready: {len(self.df)} tracks, {NUM_CLASSES} genres")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load 1 track, extract a 3s chunk, compute features

        Returns:
            tabular:     torch.FloatTensor, shape (88,)
            spectrogram: torch.FloatTensor, shape (1, 128, 130)
            label:       torch.LongTensor,  scalar
        """
        row = self.df.iloc[idx]
        track_id = row['track_id']
        genre_idx = row['genre_idx']

        # --- Load MP3 ---
        audio_path = get_audio_path(self.audio_dir, track_id)
        try:
            audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        except Exception as e:
            # Fallback: return silence + label ถ้า decode ล้มเหลว (rare)
            logger.error(f"Failed to load track {track_id}: {e}")
            audio = np.zeros(self.chunk_samples, dtype=np.float32)

        # --- Chunking Strategy ---
        audio_chunk = self._extract_chunk(audio)

        # --- Feature Extraction ---
        tabular = extract_tabular_features(audio_chunk, self.sr)
        spectrogram = extract_mel_spectrogram(audio_chunk, self.sr)

        # --- Convert to PyTorch tensors ---
        tabular_tensor = torch.from_numpy(tabular)           # (88,)
        spectrogram_tensor = torch.from_numpy(spectrogram)    # (1, 128, 130)
        label_tensor = torch.tensor(genre_idx, dtype=torch.long)

        return tabular_tensor, spectrogram_tensor, label_tensor

    def _extract_chunk(self, audio: np.ndarray) -> np.ndarray:
        """
        ตัด 3s chunk จาก full audio track

        Training:  random slice → temporal data augmentation
                   ทำให้โมเดลเห็นส่วนต่างๆ ของเพลงในแต่ละ epoch
        Val/Test:  center slice → deterministic, reproducible results

        ถ้าเพลงสั้นกว่า 3s → zero-pad ทางขวา
        """
        total_samples = len(audio)

        if total_samples < self.chunk_samples:
            # --- เพลงสั้นกว่า 3 วินาที: pad ด้วย silence ---
            padded = np.zeros(self.chunk_samples, dtype=np.float32)
            padded[:total_samples] = audio
            return padded

        if self.is_training:
            # --- Training: Random start position ---
            max_start = total_samples - self.chunk_samples
            start = np.random.randint(0, max_start + 1)
        else:
            # --- Val/Test: Center crop ---
            start = (total_samples - self.chunk_samples) // 2

        return audio[start:start + self.chunk_samples]


# ==============================================================================
# 5. HELPER: CREATE DATALOADERS
# ==============================================================================

def create_dataloaders(
    metadata_df: pd.DataFrame,
    audio_dir: Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Dict[str, DataLoader]:
    """
    สร้าง DataLoaders สำหรับทุก splits (training, validation, test)

    Returns:
        dict ของ {split_name: DataLoader}
    """
    loaders = {}

    for split in ['training', 'validation', 'test']:
        dataset = FMADataset(metadata_df, audio_dir, split=split)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'training'),   # shuffle เฉพาะ training
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,           # เร่ง CPU→GPU transfer
            drop_last=(split == 'training'), # drop incomplete batch ตอน train เท่านั้น
            persistent_workers=(num_workers > 0),  # reuse worker processes
        )
        loaders[split] = loader
        logger.info(f"[{split}] DataLoader: {len(dataset)} samples, {len(loader)} batches")

    return loaders


# ==============================================================================
# 6. VERIFICATION BLOCK
# ==============================================================================

def verify_pipeline():
    """
    ทดสอบ pipeline ทั้งหมด:
      1. โหลด metadata, แสดง genre distribution
      2. สร้าง DataLoader, ดึง 1 batch
      3. Print tensor shapes ทุกตัว
      4. Plot & save ตัวอย่าง mel spectrogram
    """
    print("=" * 72)
    print("  PHASE 1A VERIFICATION: FMA Native-Shape Data Pipeline")
    print("=" * 72)

    # --- 1. Load Metadata ---
    print("\n[STEP 1] Loading FMA metadata...")
    metadata_df = load_fma_tracks(FMA_METADATA_CSV)

    print(f"\n  Total tracks in fma_small: {len(metadata_df)}")
    print(f"  Genres ({NUM_CLASSES}):")
    for genre, idx in sorted(GENRE_TO_IDX.items(), key=lambda x: x[1]):
        count = len(metadata_df[metadata_df['genre_top'] == genre])
        print(f"    [{idx}] {genre:15s} → {count} tracks")

    # --- 2. Create Training DataLoader (single split for verification) ---
    print("\n[STEP 2] Creating training DataLoader...")
    train_dataset = FMADataset(metadata_df, FMA_AUDIO_DIR, split='training')

    # ใช้ num_workers=0 สำหรับ verification เพื่อให้ error messages ชัดเจน
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # single-process สำหรับ debug
        pin_memory=False,
    )

    # --- 3. Fetch One Batch & Print Shapes ---
    print(f"\n[STEP 3] Fetching 1 batch (batch_size={BATCH_SIZE})...")
    tabular_batch, spec_batch, label_batch = next(iter(train_loader))

    print(f"\n  ✓ Tabular features shape:  {tabular_batch.shape}")
    print(f"    Expected:                ({BATCH_SIZE}, 88)")
    print(f"    dtype:                   {tabular_batch.dtype}")

    print(f"\n  ✓ Mel spectrogram shape:   {spec_batch.shape}")
    print(f"    Expected:                ({BATCH_SIZE}, 1, {N_MELS}, {TARGET_TIME_FRAMES})")
    print(f"    dtype:                   {spec_batch.dtype}")

    print(f"\n  ✓ Label shape:             {label_batch.shape}")
    print(f"    Expected:                ({BATCH_SIZE},)")
    print(f"    dtype:                   {label_batch.dtype}")
    print(f"    Labels in batch:         {label_batch.tolist()}")
    print(f"    Genre names:             {[GENRE_LABELS[l] for l in label_batch.tolist()]}")

    # --- Shape assertions ---
    assert tabular_batch.shape == (BATCH_SIZE, 88), \
        f"Tabular shape mismatch: {tabular_batch.shape}"
    assert spec_batch.shape == (BATCH_SIZE, 1, N_MELS, TARGET_TIME_FRAMES), \
        f"Spectrogram shape mismatch: {spec_batch.shape}"
    assert label_batch.shape == (BATCH_SIZE,), \
        f"Label shape mismatch: {label_batch.shape}"

    print("\n  ✓ All shape assertions PASSED!")

    # --- 4. Plot & Save Sample Spectrogram ---
    print("\n[STEP 4] Plotting sample mel spectrogram...")
    sample_idx = 0
    sample_spec = spec_batch[sample_idx, 0].numpy()   # (128, 130)
    sample_label = GENRE_LABELS[label_batch[sample_idx].item()]

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    img = ax.imshow(
        sample_spec,
        aspect='auto',
        origin='lower',
        cmap='magma',
        extent=[0, CHUNK_DURATION, 0, SAMPLE_RATE // 2],
    )
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title(f'Log-Mel Spectrogram — Genre: {sample_label} — Shape: {MEL_SHAPE}', fontsize=13)
    fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Amplitude (dB)')
    plt.tight_layout()

    save_path = OUTPUT_DIR / "fma_native_spectrogram.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Spectrogram saved to: {save_path}")

    # --- 5. Quick Stats ---
    print("\n[STEP 5] Feature value statistics (first sample):")
    sample_tab = tabular_batch[sample_idx].numpy()
    print(f"  Tabular — min: {sample_tab.min():.4f}, max: {sample_tab.max():.4f}, "
          f"mean: {sample_tab.mean():.4f}, std: {sample_tab.std():.4f}")
    print(f"  Mel Spec — min: {sample_spec.min():.2f} dB, max: {sample_spec.max():.2f} dB, "
          f"mean: {sample_spec.mean():.2f} dB")

    # --- Check for NaN / Inf ---
    assert not torch.isnan(tabular_batch).any(), "NaN detected in tabular features!"
    assert not torch.isnan(spec_batch).any(), "NaN detected in spectrograms!"
    assert not torch.isinf(tabular_batch).any(), "Inf detected in tabular features!"
    assert not torch.isinf(spec_batch).any(), "Inf detected in spectrograms!"
    print("\n  ✓ No NaN/Inf values detected!")

    print("\n" + "=" * 72)
    print("  PHASE 1A VERIFICATION COMPLETE — Pipeline is READY")
    print("=" * 72)

    return metadata_df, train_loader


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    verify_pipeline()
