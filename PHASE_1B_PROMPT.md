# PHASE 1B PROMPT — ให้ Claude Code ในเครื่อง

คัดลอก prompt ด้านล่างทั้งหมดไปวางใน Claude Code:

---

[CONTEXT: COMPLETED PHASE 1A]
เราสร้าง real-time music genre classification pipeline สำหรับ FMA Small dataset (8000 tracks, 8 genres) เสร็จแล้วใน Phase 1A โดยมีไฟล์เหล่านี้พร้อมใช้:

Project root: D:\patt\project\pattern-music\

Files ที่มีอยู่แล้ว:
- `fma_phase1a_data_pipeline.py` — มี functions: `extract_tabular_features()` (88 features), `extract_mel_spectrogram()` (output shape 1,128,130), `get_audio_path()`, `load_fma_tracks()`, `FMADataset` class
- `output/cleaned_metadata.csv` — cleaned track metadata (track_id, genre_top, genre_idx, split columns)
- `output/tabular_scaler.pkl` — fitted sklearn StandardScaler สำหรับ 88 tabular features
- `output/scaling_stats.json` — global mel mean/std + tabular feature stats
- `output/class_weights.pt` — class weights tensor shape (8,) ≈ all ~1.0 (balanced dataset)

Config ที่ใช้:
- SAMPLE_RATE = 22050
- CHUNK_DURATION = 3.0s → CHUNK_SAMPLES = 66150
- N_MELS = 128, N_FFT = 2048, HOP_LENGTH = 512
- MEL_SHAPE = (1, 128, 130)
- TABULAR_FEATURES = 88 dimensions
- GENRES = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
- CORRUPTED_TRACK_IDS = {98565, 98567, 98569, 99134, 108925, 133297, 143992}
- Splits: FMA official train/val/test

Audio dir: D:\patt\project\pattern-music\FMA_Data\fma_small\fma_small
Metadata: D:\patt\project\pattern-music\FMA_Data\fma_metadata\fma_metadata\tracks.csv

Hardware: NVIDIA GPU (gaming laptop), Windows

[PHASE 1B TASK: PRE-EXTRACT ALL FEATURES TO .pt FILES]

ปัญหาปัจจุบัน: ทุก epoch ต้อง decode mp3 ทุกไฟล์ใหม่ → GPU idle รอ CPU → training ช้ามาก
เป้าหมาย: Extract features ครั้งเดียว save เป็น .pt files → training เร็วขึ้น 10-50x

เขียน Python script (ทั้ง .py และ .ipynb) ที่ทำ:

1. PRE-EXTRACTION SCRIPT
- อ่าน `output/cleaned_metadata.csv` (ถ้ามี) หรือ load จาก tracks.csv
- Import `extract_tabular_features` และ `extract_mel_spectrogram` จาก `fma_phase1a_data_pipeline.py`
- วนทุก track: load mp3 → extract center 3s chunk → extract ทั้ง tabular (88,) และ mel (1,128,130)
- Apply scaling: ใช้ `output/tabular_scaler.pkl` สำหรับ tabular, ใช้ `output/scaling_stats.json` สำหรับ mel normalization
- Save แต่ละ track เป็น .pt file ที่มี dict: {'tabular': tensor(88,), 'mel': tensor(1,128,130), 'label': int, 'track_id': int}
- Directory structure: `output/pt_features/{split}/track_{track_id:06d}.pt`
- แสดง progress bar (tqdm), log errors, skip corrupted tracks
- สรุปจำนวน tracks ที่ extract สำเร็จ/ล้มเหลว per split

2. FAST PYTORCH DATASET (PreExtractedFMADataset)
- Dataset class ที่ load .pt files แทน mp3
- __getitem__ แค่ `torch.load()` + apply augmentation (training only)
- Augmentation ทำบน tensor โดยตรง:
  - SpecAugment (frequency + time masking) บน mel spectrogram — p=0.5
  - Gaussian noise บน mel spectrogram — p=0.3, snr ~15-30dB equivalent
  - สำหรับ tabular: ไม่ augment (ค่า statistics ไม่ควรถูกแก้)
- Val/Test: ไม่มี augmentation

3. TRAINING-READY DATALOADER FACTORY
- Function `create_fast_dataloaders()` ที่ return dict ของ DataLoaders
- Training: WeightedRandomSampler (ถึงแม้ dataset จะ balanced อยู่แล้ว ให้ใส่ไว้เป็น option)
- num_workers=4, pin_memory=True, persistent_workers=True
- drop_last=True สำหรับ training

4. VERIFICATION BLOCK
- Load 1 batch จาก train + val
- Print shapes ทุก tensor
- Assert shapes ถูกต้อง
- เปรียบเทียบเวลา: load 1 batch จาก .pt vs จาก mp3 (FMADataset เดิม) → print speedup ratio
- Plot 4 sample spectrograms จาก .pt dataset

5. DISK SPACE ESTIMATION
- ก่อน extract: คำนวณว่าจะใช้ disk space เท่าไหร่ (ประมาณ per-track .pt size × total tracks)
- ถาม user confirm ก่อน proceed

[IMPORTANT NOTES]
- ใช้ pathlib.Path ตลอด (Windows compatibility)
- mel spectrogram ที่ save ต้องเป็น SCALED แล้ว (normalized ด้วย global mean/std)
- tabular features ที่ save ต้องเป็น SCALED แล้ว (StandardScaler applied)
- สำหรับ training augmentation: random chunk ทำไม่ได้แล้ว (เพราะ pre-extract center chunk เดียว) → ชดเชยด้วย SpecAugment + noise ที่ aggressive ขึ้นเล็กน้อย
- Training chunks: extract หลาย chunks per track (3 random positions + 1 center) เพื่อเพิ่ม training data 4x — save เป็น track_{id:06d}_chunk{0-3}.pt
- Val/Test chunks: extract เฉพาะ center chunk เดียว
- ให้ output เป็นทั้ง .py (importable module) และ .ipynb (interactive notebook)
- Comment เป็นภาษาไทย + อังกฤษผสม (technical terms เป็นอังกฤษ)

[OUTPUT STRUCTURE]
output/
├── pt_features/
│   ├── training/
│   │   ├── track_000002_chunk0.pt
│   │   ├── track_000002_chunk1.pt
│   │   ├── track_000002_chunk2.pt
│   │   ├── track_000002_chunk3.pt
│   │   └── ...
│   ├── validation/
│   │   ├── track_000005_chunk0.pt
│   │   └── ...
│   └── test/
│       ├── track_000010_chunk0.pt
│       └── ...
├── cleaned_metadata.csv        (from Phase 1A)
├── tabular_scaler.pkl          (from Phase 1A)
├── scaling_stats.json          (from Phase 1A)
└── class_weights.pt            (from Phase 1A)

[SUCCESS CRITERIA]
- ทุก .pt file load ได้ถูกต้อง, shapes match (88,) และ (1,128,130)
- Speedup ≥ 10x เทียบกับ mp3 loading
- No NaN/Inf in any tensor
- Training data ~4x จาก original (multi-chunk extraction)
- Disk usage estimation ถูกต้อง ±10%
