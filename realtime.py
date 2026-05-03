"""
Part 5: Real-time Music Genre Classification Pipeline
- รับเสียงแบบ real-time ด้วย torchaudio / sounddevice
- แบ่งเป็น chunk ทีละ N วินาที
- pad ให้ครบ 30 วินาที แล้วส่งเข้า ONNX Runtime
- Soft Voting เฉลี่ย probability ทุก chunk
- เมื่อหยุด → Majority Voting หา Final Genre
- แสดงผลแบบ real-time ผ่าน Terminal
"""

import time
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import onnxruntime as ort
import sounddevice as sd
from collections import Counter
from pathlib import Path
import threading
import queue
import sys
import os

# ==================== CONFIG ====================
SAMPLE_RATE   = 22050       # Hz
CHUNK_SEC     = 3           # วินาทีต่อ chunk
N_MELS        = 128
N_FFT         = 2048
HOP_LENGTH    = 512
DURATION      = 30          # วินาทีที่ model train มา
NUM_SAMPLES   = SAMPLE_RATE * DURATION       # 661500
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SEC      # 66150

MODEL_PATH  = "models/deep_cnn2d.onnx"      # path ไปยัง ONNX model
GENRES      = [
    "Electronic", "Experimental", "Folk", "Hip-Hop",
    "Instrumental", "International", "Pop", "Rock"
]
# ================================================

# Mel-Spectrogram transform (เหมือนตอน train)
mel_transform   = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                                   hop_length=HOP_LENGTH, n_mels=N_MELS)
amplitude_to_db = T.AmplitudeToDB(top_db=80)


def load_model(model_path: str) -> ort.InferenceSession:
    """โหลด ONNX model"""
    if not Path(model_path).exists():
        print(f"[ERROR] ไม่พบ model: {model_path}")
        sys.exit(1)
    session = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    print(f"[INFO] โหลด model สำเร็จ: {model_path}")
    provider = session.get_providers()[0]
    print(f"[INFO] ใช้ provider: {provider}")
    return session


def preprocess_chunk(waveform: np.ndarray) -> np.ndarray:
    """
    แปลง raw waveform → Mel-Spectrogram
    waveform: numpy array shape (samples,)
    return:   numpy array shape (1, 1, 128, T)
    """
    wav = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)  # (1, samples)

    # Pad ให้ครบ 30 วินาที
    if wav.shape[1] < NUM_SAMPLES:
        pad = NUM_SAMPLES - wav.shape[1]
        wav = torch.nn.functional.pad(wav, (0, pad))
    else:
        wav = wav[:, :NUM_SAMPLES]

    # Mel-Spectrogram
    mel = mel_transform(wav)        # (1, 128, T)
    mel = amplitude_to_db(mel)      # dB scale

    # Normalize (เหมือนตอน train)
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)

    return mel.unsqueeze(0).numpy()  # (1, 1, 128, T)


def inference(session: ort.InferenceSession, mel: np.ndarray) -> np.ndarray:
    """
    รัน ONNX inference
    return: probability array shape (8,)
    """
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: mel})[0]  # (1, 8)
    # Softmax
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return probs.squeeze()  # (8,)


def format_bar(prob: float, width: int = 20) -> str:
    """สร้าง progress bar จาก probability"""
    filled = int(prob * width)
    return "█" * filled + "░" * (width - filled)


def display_results(probs_avg: np.ndarray, chunk_count: int, elapsed: float):
    """แสดงผล probability แต่ละ genre แบบ real-time"""
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 55)
    print("   🎵  Real-time Music Genre Classification")
    print("=" * 55)
    print(f"   Chunks processed : {chunk_count}")
    print(f"   Time elapsed     : {elapsed:.1f}s")
    print("-" * 55)

    top_idx = int(np.argmax(probs_avg))
    for i, (genre, prob) in enumerate(zip(GENRES, probs_avg)):
        marker = " ◀ " if i == top_idx else "   "
        bar = format_bar(prob)
        print(f"   {genre:<15} {bar} {prob*100:5.1f}%{marker}")

    print("-" * 55)
    print(f"   Current prediction: {GENRES[top_idx]}")
    print("=" * 55)
    print("   กด Ctrl+C เพื่อหยุดและดูผลสรุป")


def measure_rtf(chunk_sec: float, inference_time: float) -> float:
    """
    Real-Time Factor (RTF) = inference_time / audio_duration
    RTF < 1 = เร็วกว่า real-time ✅
    """
    return inference_time / chunk_sec


def run_from_file(model_path: str, file_path: str, true_genre: str = None):
    """
    อ่านเสียงจากไฟล์แทน mic จำลองการรับเสียงแบบ real-time ทีละ chunk
    file_path:  path ไปยังไฟล์เสียง (.mp3, .wav, etc.)
    true_genre: genre จริงของเพลง (optional) สำหรับเช็ค accuracy
    """
    session = load_model(model_path)

    if not Path(file_path).exists():
        print(f"[ERROR] ไม่พบไฟล์: {file_path}")
        sys.exit(1)

    print(f"\n[INFO] โหลดไฟล์: {file_path}")
    
    # ---------------------------------------------------------
    # ใช้ soundfile อ่านแทน torchaudio.load เพื่อแก้ปัญหาบน Windows
    import soundfile as sf
    import torch
    
    audio_data, sr = sf.read(file_path)
    
    if audio_data.ndim == 1:
        waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
    else:
        waveform = torch.tensor(audio_data, dtype=torch.float32).t()
    # ---------------------------------------------------------

    # แปลงเป็น mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample ถ้า sample rate ไม่ตรง
    if sr != SAMPLE_RATE:
        waveform = T.Resample(sr, SAMPLE_RATE)(waveform)

    audio = waveform.squeeze(0).numpy()  # (samples,)
    total_samples = len(audio)
    total_chunks  = total_samples // CHUNK_SAMPLES
    print(f"[INFO] ความยาวเพลง : {total_samples / SAMPLE_RATE:.1f}s → {total_chunks} chunks")
    if true_genre:
        print(f"[INFO] Genre จริง  : {true_genre}\n")

    all_probs   = []
    rtf_list    = []
    start_time  = time.time()

    for i in range(total_chunks):
        chunk = audio[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES]

        t0    = time.perf_counter()
        mel   = preprocess_chunk(chunk)
        probs = inference(session, mel)
        t1    = time.perf_counter()

        inference_time = t1 - t0
        rtf = measure_rtf(CHUNK_SEC, inference_time)
        rtf_list.append(rtf)

        all_probs.append(probs)
        probs_avg = np.mean(all_probs, axis=0)

        display_results(probs_avg, i + 1, time.time() - start_time)
        print(f"   Inference time   : {inference_time*1000:.1f}ms  |  RTF: {rtf:.3f} {'✅' if rtf < 1 else '⚠️'}")

        # จำลอง real-time: รอให้ครบ chunk duration
        elapsed = time.perf_counter() - t0
        sleep_time = CHUNK_SEC - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # ===== FINAL SUMMARY =====
    final_probs  = np.mean(all_probs, axis=0)
    soft_genre   = GENRES[int(np.argmax(final_probs))]
    chunk_preds  = [int(np.argmax(p)) for p in all_probs]
    majority_idx = Counter(chunk_preds).most_common(1)[0][0]
    maj_genre    = GENRES[majority_idx]
    avg_rtf      = np.mean(rtf_list)

    print("\n" + "=" * 55)
    print("   🎵  FINAL RESULT")
    print("=" * 55)
    print(f"   Total chunks     : {total_chunks}")
    print(f"   Total duration   : {total_chunks * CHUNK_SEC}s")
    print(f"   Avg RTF          : {avg_rtf:.3f} {'✅ (< 1)' if avg_rtf < 1 else '⚠️ (>= 1)'}")
    print("-" * 55)
    print(f"   Soft Voting      : {soft_genre}")
    print(f"   Majority Voting  : {maj_genre}")
    print("-" * 55)
    if true_genre:
        correct = "✅ ถูก" if maj_genre.lower() == true_genre.lower() else "❌ ผิด"
        print(f"   Genre จริง       : {true_genre}")
        print(f"   ผล               : {correct}")
        print("-" * 55)
    print(f"   Final Genre      : {maj_genre}")
    print("=" * 55)

    print("\n   Per-chunk predictions:")
    for i, pred_idx in enumerate(chunk_preds):
        bar = format_bar(all_probs[i][pred_idx], width=10)
        print(f"   Chunk {i+1:02d}: {GENRES[pred_idx]:<15} {bar} {all_probs[i][pred_idx]*100:.1f}%")


def run_realtime(model_path: str = MODEL_PATH):
    """Main loop สำหรับ real-time classification"""
    session     = load_model(model_path)
    audio_queue = queue.Queue()

    # Callback รับเสียงจาก mic
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[WARN] {status}")
        audio_queue.put(indata[:, 0].copy())  # mono

    print(f"\n[INFO] เริ่มรับเสียง... (chunk = {CHUNK_SEC}s, sample_rate = {SAMPLE_RATE}Hz)")
    print("[INFO] กด Ctrl+C เพื่อหยุด\n")

    all_probs    = []     # เก็บ probability ทุก chunk สำหรับ Majority Voting
    chunk_count  = 0
    buffer       = np.array([], dtype=np.float32)
    start_time   = time.time()
    rtf_list     = []

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=1024,
                            callback=audio_callback):
            while True:
                # สะสม audio จนครบ 1 chunk
                while len(buffer) < CHUNK_SAMPLES:
                    try:
                        data = audio_queue.get(timeout=1.0)
                        buffer = np.concatenate([buffer, data])
                    except queue.Empty:
                        continue

                chunk  = buffer[:CHUNK_SAMPLES]
                buffer = buffer[CHUNK_SAMPLES:]

                # Preprocess + Inference พร้อมจับเวลา
                t0  = time.perf_counter()
                mel = preprocess_chunk(chunk)
                probs = inference(session, mel)
                t1  = time.perf_counter()

                inference_time = t1 - t0
                rtf = measure_rtf(CHUNK_SEC, inference_time)
                rtf_list.append(rtf)

                # Soft Voting: เฉลี่ย probability สะสม
                all_probs.append(probs)
                chunk_count += 1
                probs_avg = np.mean(all_probs, axis=0)

                display_results(probs_avg, chunk_count, time.time() - start_time)
                print(f"   Inference time   : {inference_time*1000:.1f}ms  |  RTF: {rtf:.3f} {'✅' if rtf < 1 else '⚠️'}")

    except KeyboardInterrupt:
        print("\n\n[INFO] หยุดรับเสียงแล้ว กำลังสรุปผล...\n")

    # ===== FINAL SUMMARY =====
    if not all_probs:
        print("[WARN] ไม่มีข้อมูลเสียง")
        return

    # Soft Voting final
    final_probs = np.mean(all_probs, axis=0)
    soft_genre  = GENRES[int(np.argmax(final_probs))]

    # Majority Voting (นับว่า chunk ไหน predict อะไรมากที่สุด)
    chunk_preds  = [int(np.argmax(p)) for p in all_probs]
    majority_idx = Counter(chunk_preds).most_common(1)[0][0]
    maj_genre    = GENRES[majority_idx]

    avg_rtf = np.mean(rtf_list) if rtf_list else 0

    print("=" * 55)
    print("   🎵  FINAL RESULT")
    print("=" * 55)
    print(f"   Total chunks     : {chunk_count}")
    print(f"   Total duration   : {chunk_count * CHUNK_SEC}s")
    print(f"   Avg RTF          : {avg_rtf:.3f} {'✅ (< 1)' if avg_rtf < 1 else '⚠️ (>= 1)'}")
    print("-" * 55)
    print(f"   Soft Voting      : {soft_genre}")
    print(f"   Majority Voting  : {maj_genre}")
    print("-" * 55)
    print(f"   Final Genre      : {maj_genre}")
    print("=" * 55)

    # Per-chunk breakdown
    print("\n   Per-chunk predictions:")
    for i, pred_idx in enumerate(chunk_preds):
        bar = format_bar(all_probs[i][pred_idx], width=10)
        print(f"   Chunk {i+1:02d}: {GENRES[pred_idx]:<15} {bar} {all_probs[i][pred_idx]*100:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-time Music Genre Classification")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to ONNX model")
    parser.add_argument("--file",  type=str, default=None, help="Path to audio file (.mp3/.wav) แทน mic")
    parser.add_argument("--genre", type=str, default=None, help="Genre จริงของเพลง (optional) เช่น Rock")
    args = parser.parse_args()

    if args.file:
        run_from_file(model_path=args.model, file_path=args.file, true_genre=args.genre)
    else:
        run_realtime(model_path=args.model)