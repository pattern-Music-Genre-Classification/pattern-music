from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import realtime


@dataclass
class BenchmarkResult:
    model: str
    status: str
    final_genre: str = "-"
    raw_majority: str = "-"
    all_window_soft: str = "-"
    windows: int = 0
    avg_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0
    avg_rtl: float = 0.0
    p95_rtl: float = 0.0
    load_sec: float = 0.0
    total_sec: float = 0.0
    top_prob: float = 0.0
    error: str = ""

    @property
    def realtime_ready(self) -> str:
        if self.status != "ok":
            return "no"
        return "yes" if self.avg_rtl < 1.0 else "no"


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct
    low = int(np.floor(rank))
    high = int(np.ceil(rank))
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def parse_selection(raw: str, count: int) -> list[int]:
    raw = raw.strip().lower()
    if raw in {"all", "a", "*"}:
        return list(range(count))

    selected: set[int] = set()
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            if not start_raw.isdigit() or not end_raw.isdigit():
                raise ValueError("Range must look like 2-5")
            start = int(start_raw)
            end = int(end_raw)
            if start > end:
                start, end = end, start
            selected.update(range(start - 1, end))
        else:
            if not part.isdigit():
                raise ValueError("Selection must be numbers, ranges, or all")
            selected.add(int(part) - 1)

    invalid = [idx for idx in selected if idx < 0 or idx >= count]
    if invalid:
        raise ValueError("Selected number is not in the list")
    return sorted(selected)


def prompt_models(model_files: Sequence[Path], base: Path) -> list[Path]:
    while True:
        print()
        print("Choose models to benchmark")
        print("--------------------------")
        for index, path in enumerate(model_files, start=1):
            print(f"{index:2d}. {realtime.display_path(path, base)}")
        print()
        print("Type examples: all | 1,3,4 | 2-5")
        raw = input("Select models: ")
        try:
            indices = parse_selection(raw, len(model_files))
        except ValueError as error:
            print(f"Invalid selection: {error}")
            continue
        if indices:
            return [model_files[idx] for idx in indices]
        print("Please select at least one model.")


def resolve_models(raw_models: str | None, models_dir: Path) -> list[Path]:
    model_files = realtime.find_files(models_dir, realtime.MODEL_EXTENSIONS)
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")

    if raw_models is None:
        return prompt_models(model_files, models_dir)

    if raw_models.strip().lower() in {"all", "a", "*"}:
        return model_files

    paths: list[Path] = []
    for raw in raw_models.split(","):
        item = raw.strip().strip('"')
        if not item:
            continue
        path = Path(item)
        if not path.exists():
            path = models_dir / item
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {item}")
        paths.append(path)
    if not paths:
        raise ValueError("No models selected")
    return paths


def resolve_song(raw_file: str | None, songs_dir: Path) -> Path:
    if raw_file:
        path = Path(raw_file)
        if not path.exists():
            path = songs_dir / raw_file
        if not path.exists():
            raise FileNotFoundError(f"Song not found: {raw_file}")
        return path

    song_files = realtime.find_files(songs_dir, realtime.AUDIO_EXTENSIONS)
    if not song_files:
        raise FileNotFoundError(f"No audio files found in {songs_dir}")
    return realtime.prompt_choice("Choose one song for benchmark", song_files, songs_dir)


def get_audio_config(args: argparse.Namespace, config: dict[str, object]) -> tuple[int, float, int, int, int, float, float | None]:
    sample_rate = int(args.sample_rate or config.get("sample_rate", realtime.SAMPLE_RATE))
    window_sec = float(args.window_sec or config.get("chunk_duration", realtime.WINDOW_SEC))
    n_mels = int(args.n_mels or config.get("n_mels", realtime.N_MELS))
    n_fft = int(args.n_fft or config.get("n_fft", realtime.N_FFT))
    mel_hop_length = int(args.mel_hop_length or config.get("hop_length", realtime.MEL_HOP_LENGTH))
    f_min = float(args.f_min if args.f_min is not None else config.get("fmin", realtime.F_MIN))
    raw_f_max = args.f_max if args.f_max is not None else config.get("fmax", realtime.F_MAX)
    f_max = None if raw_f_max is None else float(raw_f_max)
    return sample_rate, window_sec, n_mels, n_fft, mel_hop_length, f_min, f_max


def benchmark_one_model(model_path: Path, song_path: Path, args: argparse.Namespace) -> BenchmarkResult:
    result = BenchmarkResult(model=model_path.name, status="failed")
    started = time.perf_counter()

    try:
        device = realtime.choose_device(args.device)
        load_started = time.perf_counter()
        backends = realtime.load_backends([model_path], [], device, None)
        predictor = realtime.EnsemblePredictor(backends)
        result.load_sec = time.perf_counter() - load_started

        checkpoint_config = predictor.config or {}
        genres = predictor.class_names or checkpoint_config.get("genres") or realtime.DEFAULT_GENRES
        genres = [str(genre) for genre in genres]

        sample_rate, window_sec, n_mels, n_fft, mel_hop_length, f_min, f_max = get_audio_config(args, checkpoint_config)
        preprocessor = realtime.AudioPreprocessor(
            sample_rate=sample_rate,
            window_sec=window_sec,
            n_mels=n_mels,
            n_fft=n_fft,
            mel_hop_length=mel_hop_length,
            f_min=f_min,
            f_max=f_max,
            input_format=args.input_format,
        )

        hop_samples = int(round(sample_rate * args.hop_sec))
        if hop_samples <= 0:
            raise ValueError("--hop-sec must be positive")

        realtime.validate_class_count(genres, predictor, preprocessor)
        audio = realtime.load_audio_file(song_path, sample_rate)

        vote_buffer: deque[np.ndarray] = deque(maxlen=args.vote_windows)
        latest_history: list[np.ndarray] = []
        soft_history: list[np.ndarray] = []
        soft_decisions: list[int] = []
        inference_times: list[float] = []

        for window_index, _, window in realtime.iter_file_windows(
            audio,
            preprocessor.window_samples,
            hop_samples,
            include_tail=not args.no_tail,
        ):
            if args.max_windows and window_index > args.max_windows:
                break
            probs, inference_sec = realtime.process_window(predictor, preprocessor, window)
            latest_history.append(probs)
            inference_times.append(inference_sec)
            vote_buffer.append(probs)
            soft_probs = np.mean(vote_buffer, axis=0)
            soft_history.append(soft_probs)
            soft_decisions.append(int(np.argmax(soft_probs)))

        if not latest_history:
            raise RuntimeError("No complete window was processed")

        latest_mean = np.mean(latest_history, axis=0)
        soft_mean = np.mean(soft_history, axis=0)
        final_idx = realtime.majority_vote(soft_decisions, soft_mean)
        raw_idx = realtime.majority_vote([int(np.argmax(probs)) for probs in latest_history], latest_mean)
        all_window_idx = int(np.argmax(latest_mean))

        avg_sec = float(statistics.mean(inference_times))
        median_sec = float(statistics.median(inference_times))
        p95_sec = percentile(inference_times, 0.95)
        max_sec = float(max(inference_times))
        hop_sec = hop_samples / sample_rate

        result.status = "ok"
        result.final_genre = genres[final_idx]
        result.raw_majority = genres[raw_idx]
        result.all_window_soft = genres[all_window_idx]
        result.windows = len(latest_history)
        result.avg_ms = avg_sec * 1000
        result.median_ms = median_sec * 1000
        result.p95_ms = p95_sec * 1000
        result.max_ms = max_sec * 1000
        result.avg_rtl = avg_sec / hop_sec
        result.p95_rtl = p95_sec / hop_sec
        result.top_prob = float(soft_mean[final_idx])
        result.total_sec = time.perf_counter() - started
        return result

    except Exception as error:
        result.error = str(error)
        result.total_sec = time.perf_counter() - started
        return result
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def print_progress_result(index: int, total: int, result: BenchmarkResult) -> None:
    prefix = f"[{index}/{total}] {result.model}"
    if result.status != "ok":
        print(f"{prefix}: FAILED - {result.error}")
        return
    print(
        f"{prefix}: final={result.final_genre}, "
        f"avg={result.avg_ms:.1f}ms, RTL={result.avg_rtl:.3f}, "
        f"ready={result.realtime_ready}"
    )


def print_summary(results: Sequence[BenchmarkResult], song_path: Path, hop_sec: float) -> None:
    ok_results = [result for result in results if result.status == "ok"]
    failed_results = [result for result in results if result.status != "ok"]
    ranked = sorted(ok_results, key=lambda item: (item.avg_rtl >= 1.0, item.avg_rtl, item.p95_rtl))

    print()
    print("=" * 118)
    print("Realtime Model Benchmark Summary")
    print("=" * 118)
    print(f"Song     : {song_path}")
    print(f"RTL      : average_inference_time / hop_time, hop_time={hop_sec:g}s")
    print(f"Criterion: RTL < 1 means the model can keep up with realtime updates")
    print("-" * 118)
    header = (
        f"{'#':>2}  {'Model':<32} {'Final':<15} {'Win':>4} "
        f"{'Avg ms':>9} {'P95 ms':>9} {'Max ms':>9} {'RTL':>7} {'P95 RTL':>8} {'Ready':>6} {'Prob':>7}"
    )
    print(header)
    print("-" * 118)
    for rank, result in enumerate(ranked, start=1):
        print(
            f"{rank:>2}  {result.model:<32} {result.final_genre:<15} {result.windows:>4} "
            f"{result.avg_ms:>9.1f} {result.p95_ms:>9.1f} {result.max_ms:>9.1f} "
            f"{result.avg_rtl:>7.3f} {result.p95_rtl:>8.3f} {result.realtime_ready:>6} {result.top_prob * 100:>6.1f}%"
        )

    if failed_results:
        print("-" * 118)
        print("Failed models:")
        for result in failed_results:
            print(f"- {result.model}: {result.error}")

    if ranked:
        fastest_ready = next((result for result in ranked if result.avg_rtl < 1.0), None)
        if fastest_ready:
            print("-" * 118)
            print(
                f"Recommended realtime model by speed: {fastest_ready.model} "
                f"(RTL={fastest_ready.avg_rtl:.3f}, final={fastest_ready.final_genre})"
            )
        else:
            print("-" * 118)
            print("No selected model reached RTL < 1 on this machine/settings.")
    print("=" * 118)


def write_csv(path: Path, results: Sequence[BenchmarkResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "status",
                "final_genre",
                "raw_majority",
                "all_window_soft",
                "windows",
                "avg_ms",
                "median_ms",
                "p95_ms",
                "max_ms",
                "avg_rtl",
                "p95_rtl",
                "realtime_ready",
                "top_prob",
                "load_sec",
                "total_sec",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model": result.model,
                    "status": result.status,
                    "final_genre": result.final_genre,
                    "raw_majority": result.raw_majority,
                    "all_window_soft": result.all_window_soft,
                    "windows": result.windows,
                    "avg_ms": f"{result.avg_ms:.6f}",
                    "median_ms": f"{result.median_ms:.6f}",
                    "p95_ms": f"{result.p95_ms:.6f}",
                    "max_ms": f"{result.max_ms:.6f}",
                    "avg_rtl": f"{result.avg_rtl:.6f}",
                    "p95_rtl": f"{result.p95_rtl:.6f}",
                    "realtime_ready": result.realtime_ready,
                    "top_prob": f"{result.top_prob:.6f}",
                    "load_sec": f"{result.load_sec:.6f}",
                    "total_sec": f"{result.total_sec:.6f}",
                    "error": result.error,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark multiple realtime genre models on one song",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", help="Comma-separated model paths/names, or 'all'. If omitted, open a menu.")
    parser.add_argument("--file", help="Song path/name. If omitted, open a menu from songs/.")
    parser.add_argument("--models-dir", default="models", help="Folder containing candidate models.")
    parser.add_argument("--songs-dir", default="songs", help="Folder containing benchmark songs.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hop-sec", type=float, default=realtime.HOP_SEC)
    parser.add_argument("--vote-windows", type=int, default=realtime.VOTE_WINDOWS)
    parser.add_argument("--max-windows", type=int, help="Limit number of windows for a quick benchmark.")
    parser.add_argument("--no-tail", action="store_true", help="Do not benchmark the final non-aligned file window.")
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--window-sec", type=float)
    parser.add_argument("--n-mels", type=int)
    parser.add_argument("--n-fft", type=int)
    parser.add_argument("--mel-hop-length", type=int)
    parser.add_argument("--f-min", type=float)
    parser.add_argument("--f-max", type=float)
    parser.add_argument("--input-format", choices=("mel4d", "mel3d", "waveform"), default="mel4d")
    parser.add_argument("--csv", help="Optional CSV output path for benchmark results.")
    args = parser.parse_args()

    if args.vote_windows <= 0:
        raise ValueError("--vote-windows must be positive")

    root = Path(__file__).resolve().parent
    models_dir = Path(args.models_dir)
    songs_dir = Path(args.songs_dir)
    if not models_dir.is_absolute():
        models_dir = root / models_dir
    if not songs_dir.is_absolute():
        songs_dir = root / songs_dir

    model_paths = resolve_models(args.models, models_dir)
    song_path = resolve_song(args.file, songs_dir)

    print("=" * 90)
    print("Realtime Model Benchmark")
    print("=" * 90)
    print(f"Song       : {song_path}")
    print(f"Models     : {len(model_paths)} selected")
    print(f"Hop        : {args.hop_sec:g}s")
    print(f"RTL metric : average_inference_time / hop_time")
    if args.max_windows:
        print(f"Windows    : first {args.max_windows} windows only")
    print("=" * 90)

    results: list[BenchmarkResult] = []
    for index, model_path in enumerate(model_paths, start=1):
        result = benchmark_one_model(model_path, song_path, args)
        results.append(result)
        print_progress_result(index, len(model_paths), result)

    print_summary(results, song_path, args.hop_sec)

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = root / csv_path
        write_csv(csv_path, results)
        print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(130)
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
