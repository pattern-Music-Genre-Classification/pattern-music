"""
Part 1: Data Preparation
- โหลด metadata จาก FMA dataset
- กรองเฉพาะ fma_small (8 genres)
- ตรวจสอบไฟล์เสียงว่ามีครบและอ่านได้
- แบ่ง train/val/test split
- บันทึก CSV สำหรับใช้ใน Part 2
"""

import os
import ast
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==================== CONFIG ====================
DATA_DIR = Path("data")
AUDIO_DIR = DATA_DIR / "fma_small"
METADATA_DIR = DATA_DIR / "fma_metadata"
OUTPUT_DIR = DATA_DIR / "processed"

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1
RANDOM_SEED = 42
# ================================================


def load_metadata(metadata_dir: Path) -> pd.DataFrame:
    """โหลดและ parse metadata จากไฟล์ tracks.csv ของ FMA"""
    tracks_path = metadata_dir / "tracks.csv"
    print(f"[1/4] Loading metadata from {tracks_path} ...")

    # tracks.csv ของ FMA มี multi-level header 2 แถว
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])

    # ดึงเฉพาะคอลัมน์ที่ต้องการ
    subset = tracks["set", "subset"]          # small / medium / large
    genre  = tracks["track", "genre_top"]     # genre label

    df = pd.DataFrame({
        "track_id": tracks.index,
        "subset":   subset.values,
        "genre":    genre.values,
    })

    print(f"    Total tracks in metadata: {len(df)}")
    return df


def filter_small(df: pd.DataFrame) -> pd.DataFrame:
    """กรองเฉพาะ fma_small และ drop แถวที่ไม่มี genre"""
    df_small = df[df["subset"] == "small"].copy()
    df_small = df_small.dropna(subset=["genre"])
    df_small["genre"] = df_small["genre"].astype(str).str.strip()
    print(f"[2/4] After filtering fma_small: {len(df_small)} tracks, "
          f"{df_small['genre'].nunique()} genres")
    print(f"    Genres: {sorted(df_small['genre'].unique())}")
    return df_small


def build_audio_path(track_id: int, audio_dir: Path) -> Path:
    """FMA เก็บไฟล์ใน subfolder จาก 3 ตัวแรกของ track_id เช่น 000/000002.mp3"""
    tid_str = f"{track_id:06d}"
    return audio_dir / tid_str[:3] / f"{tid_str}.mp3"


def verify_audio_files(df: pd.DataFrame, audio_dir: Path) -> pd.DataFrame:
    """ตรวจว่าไฟล์ .mp3 มีอยู่จริง และเพิ่ม column 'audio_path'"""
    print(f"[3/4] Verifying audio files in {audio_dir} ...")

    paths = df["track_id"].apply(lambda tid: build_audio_path(tid, audio_dir))
    exists = paths.apply(lambda p: p.exists())

    df = df.copy()
    df["audio_path"] = paths.astype(str)
    df["file_exists"] = exists

    missing = (~exists).sum()
    print(f"    Found:   {exists.sum()} files")
    print(f"    Missing: {missing} files")

    if missing > 0:
        print("    ⚠ Missing track IDs (first 10):",
              df.loc[~exists, "track_id"].head(10).tolist())

    # เก็บเฉพาะไฟล์ที่มีอยู่จริง
    df = df[exists].reset_index(drop=True)
    return df


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """แบ่ง stratified train / val / test"""
    print(f"[4/4] Splitting dataset (train={TRAIN_RATIO}, "
          f"val={VAL_RATIO}, test={TEST_RATIO}) ...")

    # แบ่ง train vs (val+test) ก่อน
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["genre"],
        random_state=RANDOM_SEED,
    )

    # แบ่ง val vs test จาก temp
    val_ratio_adj = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adj),
        stratify=temp_df["genre"],
        random_state=RANDOM_SEED,
    )

    print(f"    Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def encode_labels(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """แปลง genre string → integer label"""
    genres = sorted(train_df["genre"].unique())
    genre2idx = {g: i for i, g in enumerate(genres)}
    idx2genre = {i: g for g, i in genre2idx.items()}

    for df in [train_df, val_df, test_df]:
        df["label"] = df["genre"].map(genre2idx)

    print(f"\n    Label mapping:")
    for g, i in genre2idx.items():
        print(f"      {i}: {g}")

    return train_df, val_df, test_df, idx2genre


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """บันทึก CSV แต่ละ split"""
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = ["track_id", "audio_path", "genre", "label"]

    train_df[cols].to_csv(output_dir / "train.csv", index=False)
    val_df[cols].to_csv(output_dir  / "val.csv",   index=False)
    test_df[cols].to_csv(output_dir / "test.csv",  index=False)

    print(f"\n✅ Saved splits to {output_dir}/")
    print(f"   train.csv  → {len(train_df)} rows")
    print(f"   val.csv    → {len(val_df)} rows")
    print(f"   test.csv   → {len(test_df)} rows")


def main():
    print("=" * 50)
    print("  Part 1: Data Preparation")
    print("=" * 50)

    df = load_metadata(METADATA_DIR)
    df = filter_small(df)
    df = verify_audio_files(df, AUDIO_DIR)

    train_df, val_df, test_df = split_dataset(df)
    train_df, val_df, test_df, idx2genre = encode_labels(train_df, val_df, test_df)

    save_splits(train_df, val_df, test_df, OUTPUT_DIR)

    # บันทึก label mapping ไว้ใช้ต่อใน Part 2-5
    label_map = pd.DataFrame(
        list(idx2genre.items()), columns=["label", "genre"]
    ).sort_values("label")
    label_map.to_csv(OUTPUT_DIR / "label_map.csv", index=False)
    print(f"   label_map.csv → {OUTPUT_DIR / 'label_map.csv'}")

    print("\n🎉 Data Preparation complete! ต่อไปรัน Part 2: preprocess.py")


if __name__ == "__main__":
    main()