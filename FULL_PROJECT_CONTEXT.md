# 🎵 COMPLETE PROJECT CONTEXT: Real-Time Music Genre Classification Pipeline

## PROJECT OVERVIEW

สร้าง end-to-end, real-time music genre classification system ที่ทำงานแบบ local, pure-Python streaming application ใช้ FMA Small dataset

---

## ARCHITECTURE DECISIONS (ตกลงกันแล้ว)

### Dataset
- **FMA Small** — 8,000 tracks, 8 genres, ~30s per track
- Genres: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock
- Dataset มี class balance ดีอยู่แล้ว (class weights ≈ 1.0 ทุก genre)
- ใช้ FMA official train/val/test splits

### Audio Config
- SAMPLE_RATE = 22050 (FMA native)
- CHUNK_DURATION = 3.0 seconds → CHUNK_SAMPLES = 66,150
- N_MELS = 128, N_FFT = 2048, HOP_LENGTH = 512
- **Mel Spectrogram shape: (1, 128, 130)** — NATIVE shape, ห้าม resize/interpolate เป็น 224x224

### Feature Extraction
- **Tabular features (88 dims):** MFCCs (20 × mean/std = 40), Spectral Centroid (2), Bandwidth (2), Rolloff (2), Chroma STFT (12 × 2 = 24), ZCR (2), RMS (2), Spectral Contrast (7 × 2 = 14)
- **Mel Spectrogram (1, 128, 130):** log-scaled, power_to_db with ref=np.max

### Hardware Target
- **NVIDIA GPU** (gaming laptop, RTX 4060)
- Windows OS
- CPU-first design ที่ทำงานบน GPU ได้

### Real-Time Inference Design
```
Audio Stream → [3s window, hop 1s, overlap 67%] → CNN inference → softmax probs
                                                                      ↓
                                                        Soft Vote (running avg)
                                                        → แสดงผล real-time ทุก 1 วินาที
                                                                      ↓
                                                        เพลงจบ → Majority Vote → Final Genre
```

### Model Lineup (4 models เปรียบเทียบ)
1. **Baseline 1 — Random Forest** (tabular 88 features): พิสูจน์ว่า simple statistical features พอไหม
2. **Baseline 2 — Custom 2D CNN** (mel spectrogram): lightweight from-scratch neural network benchmark
3. **SOTA Contender A — MobileNetV2** (Transfer Learning): designed for edge/mobile, treats spectrogram as static image, fastest inference
4. **SOTA Contender B — CRNN** (CNN + LSTM/GRU): CNN extracts spatial features → RNN analyzes temporal changes over time

### Evaluation Metrics
- F1-score (primary)
- Confusion Matrix
- Latency / Real-Time Factor (RTF < 1)

---

## FILE STRUCTURE

```
D:\patt\project\pattern-music\
│
├── FMA_Data\
│   ├── fma_small\fma_small\          ← Audio files (000/, 001/, ... subdirs with .mp3)
│   └── fma_metadata\fma_metadata\
│       └── tracks.csv                 ← Multi-level header CSV (header=[0,1])
│
├── fma_phase1a_data_pipeline.py       ← Core library (importable module, ไม่ต้องรัน)
├── fma_phase1a_data_pipeline.ipynb    ← Phase 1A notebook ✅ DONE
├── fma_phase1a_eda.ipynb              ← EDA notebook (optional, ยังไม่ได้รัน)
├── fma_phase1a_complete.ipynb         ← Phase 1A Complete ✅ DONE
│
└── output\
    ├── cleaned_metadata.csv           ← Cleaned track metadata ✅
    ├── skip_track_ids.txt             ← Track IDs to skip ✅
    ├── tabular_scaler.pkl             ← Fitted StandardScaler (88 features) ✅
    ├── scaling_stats.json             ← Global mel mean/std + tabular stats ✅
    ├── class_weights.pt               ← Class weights tensor (8,) ≈ all ~1.0 ✅
    ├── augmentation_comparison.png    ✅
    ├── scaling_verification.png       ✅
    ├── class_imbalance_handling.png   ✅
    └── train_vs_val_spectrograms.png  ✅
```

---

## WHAT'S BEEN DONE (Phase 1A ✅ COMPLETE — 9/9 Steps)

### 1. Data Collection ✅
- FMA Small downloaded and accessible

### 2. Data Cleaning ✅
- Corrupted tracks removed: {98565, 98567, 98569, 99134, 108925, 133297, 143992}
- Additional silent/short tracks identified via EDA scan
- `cleaned_metadata.csv` generated

### 3. EDA ✅
- Duration distribution analyzed (~30s median per track)
- Class balance verified (nearly balanced, weights ≈ 1.0)
- Feature correlation heatmap generated
- ANOVA F-score feature importance computed
- Mel spectrogram dB statistics computed

### 4. Feature Engineering ✅
- 88 tabular features extracted (MFCC, spectral, chroma, ZCR, RMS, contrast)
- Mel spectrogram (1, 128, 130) — native shape

### 5. Data Transformation ✅
- Log-mel scaling (power_to_db)
- StandardScaler for tabular features
- Global normalization for mel spectrograms

### 6. Data Augmentation ✅ (defined, applied on-the-fly)
- **AudioAugmentor class:** Gaussian Noise (p=0.3, SNR 15-30dB), Time Stretch (p=0.2, ±10%), Pitch Shift (p=0.2, ±2 semitones)
- **SpecAugment class:** Frequency masking (2 masks, max_width=15) + Time masking (2 masks, max_width=20), p=0.5
- Training only — val/test ไม่มี augmentation

### 7. Data Splitting ✅
- FMA official train/val/test splits

### 8. Feature Scaling ✅
- Tabular: StandardScaler fit on training set only → saved `tabular_scaler.pkl`
- Mel: Global mean/std from training set → saved `scaling_stats.json`
- Mel mean ≈ (check scaling_stats.json), Mel std ≈ (check scaling_stats.json)

### 9. Class Imbalance Handling ✅
- WeightedRandomSampler implemented (optional, dataset already balanced)
- Class weights computed and saved `class_weights.pt`
- Values ≈ tensor([0.999, 0.999, 0.999, 1.002, 0.999, 1.004, 0.998, 0.999])

---

## KEY CODE IN fma_phase1a_data_pipeline.py

```python
# Functions available for import:
from fma_phase1a_data_pipeline import (
    extract_tabular_features,    # (audio_1d, sr) → np.ndarray (88,)
    extract_mel_spectrogram,     # (audio_1d, sr) → np.ndarray (1, 128, 130)
    load_fma_tracks,             # (csv_path) → pd.DataFrame
    get_audio_path,              # (audio_dir, track_id) → Path
    FMADataset,                  # PyTorch Dataset (mp3-based, slow)
    create_dataloaders,          # returns dict of DataLoaders
)

# Constants:
SAMPLE_RATE = 22050
CHUNK_DURATION = 3.0
CHUNK_SAMPLES = 66150
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_TIME_FRAMES = 130
BATCH_SIZE = 32
GENRE_LABELS = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop',
                'Instrumental', 'International', 'Pop', 'Rock']
GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(GENRE_LABELS)}
NUM_CLASSES = 8
CORRUPTED_TRACK_IDS = {98565, 98567, 98569, 99134, 108925, 133297, 143992}
```

---

## AUGMENTATION CLASSES (from Phase 1A-Complete)

```python
class AudioAugmentor:
    # Methods: add_gaussian_noise(audio, snr_db), time_stretch(audio, rate),
    #          pitch_shift(audio, n_steps), apply(audio, p_noise, p_stretch, p_pitch)

class SpecAugment:
    # freq_mask_param=15, time_mask_param=20, n_freq_masks=2, n_time_masks=2
    # Method: apply(mel_spec) → augmented mel_spec (same shape)
```

---

## VERIFIED OUTPUTS (from Phase 1A runs)

```
BATCH SHAPE VERIFICATION:
  Tabular features: torch.Size([32, 88])     ✅
  Mel spectrogram:  torch.Size([32, 1, 128, 130])  ✅
  Labels:           torch.Size([32])          ✅
  No NaN/Inf detected                        ✅
  All assertions PASSED                       ✅
```

---

## FULL PHASE ROADMAP

| Phase | Task | Status |
|-------|------|--------|
| **1A** | Data pipeline + metadata + feature extraction | ✅ DONE |
| **1A-EDA** | EDA + cleaning + analysis | ✅ DONE |
| **1A-Complete** | Augmentation + scaling + class imbalance | ✅ DONE |
| **1B** | Pre-extract all features to .pt files (10-50x speedup) | ⬅️ NEXT |
| **2A** | Random Forest baseline (tabular features) | TODO |
| **2B** | Custom 2D CNN baseline (mel spectrograms) | TODO |
| **3A** | MobileNetV2 transfer learning (SOTA-A) | TODO |
| **3B** | CRNN — CNN + LSTM/GRU (SOTA-B) | TODO |
| **4** | Real-time streaming app (soft vote → majority vote) | TODO |

---

## PHASE 1B TASK: PRE-EXTRACT ALL FEATURES TO .pt FILES

ปัญหาปัจจุบัน: ทุก epoch ต้อง decode mp3 ทุกไฟล์ใหม่ → GPU idle รอ CPU → training ช้ามาก
เป้าหมาย: Extract features ครั้งเดียว save เป็น .pt files → training เร็วขึ้น 10-50x

เขียน Python script (ทั้ง .py และ .ipynb) ที่ทำ:

### 1. PRE-EXTRACTION SCRIPT
- อ่าน `output/cleaned_metadata.csv` (ถ้ามี) หรือ load จาก tracks.csv
- Import `extract_tabular_features` และ `extract_mel_spectrogram` จาก `fma_phase1a_data_pipeline.py`
- วนทุก track: load mp3 → extract chunk → extract ทั้ง tabular (88,) และ mel (1,128,130)
- Apply scaling: ใช้ `output/tabular_scaler.pkl` สำหรับ tabular, ใช้ `output/scaling_stats.json` สำหรับ mel normalization
- Save แต่ละ track เป็น .pt file ที่มี dict: {'tabular': tensor(88,), 'mel': tensor(1,128,130), 'label': int, 'track_id': int}
- Directory structure: `output/pt_features/{split}/track_{track_id:06d}.pt`
- แสดง progress bar (tqdm), log errors, skip corrupted tracks
- สรุปจำนวน tracks ที่ extract สำเร็จ/ล้มเหลว per split

### 2. TRAINING CHUNKS STRATEGY
- **Training tracks:** extract 4 chunks per track (3 random positions + 1 center) เพื่อเพิ่ม training data ~4x → save เป็น track_{id:06d}_chunk{0-3}.pt
- **Val/Test tracks:** extract เฉพาะ center chunk เดียว → track_{id:06d}_chunk0.pt
- เหตุผล: Pre-extract แล้วทำ random chunk ไม่ได้ → ชดเชยด้วย multi-chunk + SpecAugment

### 3. FAST PYTORCH DATASET (PreExtractedFMADataset)
- Dataset class ที่ load .pt files แทน mp3
- __getitem__ แค่ `torch.load()` + apply augmentation (training only)
- Augmentation ทำบน tensor โดยตรง:
  - SpecAugment (frequency + time masking) บน mel spectrogram — p=0.5
  - Gaussian noise บน mel spectrogram — p=0.3
  - สำหรับ tabular: ไม่ augment
- Val/Test: ไม่มี augmentation

### 4. TRAINING-READY DATALOADER FACTORY
- Function `create_fast_dataloaders()` ที่ return dict ของ DataLoaders
- Training: WeightedRandomSampler (optional, dataset balanced)
- num_workers=4, pin_memory=True, persistent_workers=True
- drop_last=True สำหรับ training

### 5. VERIFICATION BLOCK
- Load 1 batch จาก train + val
- Print shapes ทุก tensor
- Assert shapes ถูกต้อง
- เปรียบเทียบเวลา: load 1 batch จาก .pt vs จาก mp3 → print speedup ratio
- Plot 4 sample spectrograms จาก .pt dataset

### 6. DISK SPACE ESTIMATION
- ก่อน extract: คำนวณว่าจะใช้ disk space เท่าไหร่
- ถาม user confirm ก่อน proceed

### OUTPUT STRUCTURE
```
output/
├── pt_features/
│   ├── training/
│   │   ├── track_000002_chunk0.pt    (center)
│   │   ├── track_000002_chunk1.pt    (random)
│   │   ├── track_000002_chunk2.pt    (random)
│   │   ├── track_000002_chunk3.pt    (random)
│   │   └── ...
│   ├── validation/
│   │   ├── track_000005_chunk0.pt    (center only)
│   │   └── ...
│   └── test/
│       ├── track_000010_chunk0.pt    (center only)
│       └── ...
├── cleaned_metadata.csv        (from Phase 1A)
├── tabular_scaler.pkl          (from Phase 1A)
├── scaling_stats.json          (from Phase 1A)
└── class_weights.pt            (from Phase 1A)
```

### SUCCESS CRITERIA
- ทุก .pt file load ได้ถูกต้อง, shapes match (88,) และ (1,128,130)
- Speedup ≥ 10x เทียบกับ mp3 loading
- No NaN/Inf in any tensor
- Training data ~4x จาก original (multi-chunk)
- Disk usage estimation ถูกต้อง ±10%

### IMPORTANT NOTES
- ใช้ pathlib.Path ตลอด (Windows compatibility)
- mel spectrogram ที่ save ต้องเป็น SCALED แล้ว (normalized ด้วย global mean/std)
- tabular features ที่ save ต้องเป็น SCALED แล้ว (StandardScaler applied)
- Comment เป็นภาษาไทย + อังกฤษผสม (technical terms เป็นอังกฤษ)
- ให้ output เป็นทั้ง .py (importable module) และ .ipynb (interactive notebook)
