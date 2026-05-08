from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import os
import queue
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.serialization import safe_globals

try:
    import onnxruntime as ort
except ImportError:  # Optional: only needed for .onnx models.
    ort = None

try:
    import sounddevice as sd
except ImportError:  # Optional: only needed for microphone mode.
    sd = None

try:
    import soundfile as sf
except ImportError:  # Optional fallback handled in load_audio_file().
    sf = None


SAMPLE_RATE = 22050
WINDOW_SEC = 15.0
HOP_SEC = 1.0
VOTE_WINDOWS = 3

N_MELS = 128
N_FFT = 2048
MEL_HOP_LENGTH = 512
F_MIN = 0.0
F_MAX = None

DEFAULT_GENRES = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock",
]

DEFAULT_MODEL_CANDIDATES = [
    Path("models/cnn_specaug_best.pt"),
    Path("models/deep_cnn2d.onnx"),
    Path(r"C:\Y3T2\PATTERN\pattern-music\models\cnn_specaug_best.pt"),
    Path(r"C:\Y3T2\PATTERN\pattern-music\models\deep_cnn2d.onnx"),
]

MODEL_EXTENSIONS = (".pt", ".pth", ".ckpt", ".onnx")
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac")

OPTIONAL_MODEL_DEPENDENCIES = {
    "timm": (
        "This usually happens with EfficientNet/timm models saved as full PyTorch models. "
        "Install it with: pip install timm"
    ),
    "torchvision": (
        "This usually happens with torchvision models saved as full PyTorch models. "
        "Install it with: pip install torchvision"
    ),
    "transformers": (
        "This is needed for ASTForAudioClassification checkpoints. "
        "Install it with: pip install transformers"
    ),
}


def resolve_default_model() -> Path:
    for path in DEFAULT_MODEL_CANDIDATES:
        if path.exists():
            return path
    return DEFAULT_MODEL_CANDIDATES[0]


def parse_genres(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    genres = [item.strip() for item in raw.split(",") if item.strip()]
    return genres or None


def parse_mic_device(raw: str | None) -> int | str | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return int(raw) if raw.isdigit() else raw


def read_label_map(path: str | Path | None) -> list[str] | None:
    if path is None:
        return None

    label_path = Path(path)
    if not label_path.exists():
        raise FileNotFoundError(f"Label map not found: {label_path}")

    with label_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Label map is empty: {label_path}")

    if "genre" not in rows[0]:
        raise ValueError("Label map must contain a 'genre' column")

    if "label" in rows[0]:
        rows = sorted(rows, key=lambda row: int(row["label"]))

    return [row["genre"] for row in rows]


def extract_class_names(obj: object) -> list[str] | None:
    if not isinstance(obj, dict):
        return None

    config = obj.get("config")
    if isinstance(config, dict):
        class_names = extract_class_names(config)
        if class_names:
            return class_names

    for key in ("class_names", "classes", "genres"):
        value = obj.get(key)
        if isinstance(value, (list, tuple)) and value:
            return [str(item) for item in value]

    for key in ("idx_to_genre", "idx_to_class", "label_map"):
        value = obj.get(key)
        if isinstance(value, dict) and value:
            try:
                return [str(value[idx]) for idx in sorted(value, key=lambda item: int(item))]
            except (TypeError, ValueError):
                return [str(value[idx]) for idx in sorted(value)]

    for key in ("genre_to_idx", "class_to_idx"):
        value = obj.get(key)
        if isinstance(value, dict) and value:
            return [str(genre) for genre, _ in sorted(value.items(), key=lambda item: int(item[1]))]

    return None


def extract_config(obj: object) -> dict[str, object] | None:
    if isinstance(obj, dict) and isinstance(obj.get("config"), dict):
        return dict(obj["config"])
    return None


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def format_bar(prob: float, width: int = 24) -> str:
    prob = float(np.clip(prob, 0.0, 1.0))
    filled = int(round(prob * width))
    return "#" * filled + "." * (width - filled)


def as_probs(logits_or_probs: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(logits_or_probs, torch.Tensor):
        values = logits_or_probs.detach().float().cpu().numpy()
    else:
        values = np.asarray(logits_or_probs, dtype=np.float32)

    values = np.squeeze(values)
    if values.ndim > 1:
        values = values[0]

    if values.ndim != 1:
        raise ValueError(f"Expected a 1D class vector, got shape {values.shape}")

    values = values.astype(np.float64)
    total = float(values.sum())
    if np.all(values >= 0.0) and np.all(values <= 1.0) and total > 0.0 and abs(total - 1.0) < 1e-3:
        return (values / total).astype(np.float32)

    values = values - np.max(values)
    exp = np.exp(values)
    return (exp / exp.sum()).astype(np.float32)


class AudioPreprocessor:
    def __init__(
        self,
        sample_rate: int,
        window_sec: float,
        n_mels: int,
        n_fft: int,
        mel_hop_length: int,
        f_min: float,
        f_max: float | None,
        input_format: str,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.window_samples = int(round(sample_rate * window_sec))
        self.input_format = input_format
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=mel_hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80)

    def __call__(self, waveform: np.ndarray) -> torch.Tensor:
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if waveform.size < self.window_samples:
            waveform = np.pad(waveform, (0, self.window_samples - waveform.size))
        elif waveform.size > self.window_samples:
            waveform = waveform[-self.window_samples :]

        wav = torch.from_numpy(waveform.copy()).float().unsqueeze(0)
        if self.input_format == "waveform":
            return wav

        mel = self.mel_transform(wav)
        mel = self.amplitude_to_db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)

        if self.input_format == "mel3d":
            return mel
        if self.input_format == "mel4d":
            return mel.unsqueeze(0)

        raise ValueError(f"Unsupported input format: {self.input_format}")


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpecAugCnn(nn.Module):
    """CNN architecture matching cnn_specaug_best.pt state_dict keys."""

    def __init__(self, num_classes: int = 8, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBnRelu(1, 32),
            nn.MaxPool2d(2),
            ConvBnRelu(32, 64),
            nn.MaxPool2d(2),
            ConvBnRelu(64, 128),
            nn.MaxPool2d(2),
            ConvBnRelu(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class CrnnGenreClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBnRelu(1, 32),
            nn.MaxPool2d(2),
            ConvBnRelu(32, 64),
            nn.MaxPool2d(2),
            ConvBnRelu(64, 128),
            nn.MaxPool2d(2),
            ConvBnRelu(128, 128),
            nn.MaxPool2d(2),
        )
        self.gru = nn.GRU(
            input_size=1024,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2).flatten(2)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class SmallResNetGenreClassifier(nn.Module):
    def __init__(self, num_classes: int = 8, dropout: float = 0.3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(ResidualBlock(32, 32), ResidualBlock(32, 32))
        self.stage2 = nn.Sequential(ResidualBlock(32, 64, stride=2), ResidualBlock(64, 64))
        self.stage3 = nn.Sequential(ResidualBlock(64, 128, stride=2), ResidualBlock(128, 128))
        self.stage4 = nn.Sequential(ResidualBlock(128, 256, stride=2), ResidualBlock(256, 256))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.classifier(x)


def build_builtin_crnn(
    state_dict: dict[str, torch.Tensor],
    checkpoint_config: dict[str, object] | None,
    genres: Sequence[str] | None,
) -> nn.Module | None:
    crnn_keys = {
        "cnn.0.block.0.weight",
        "cnn.6.block.0.weight",
        "gru.weight_ih_l0",
        "classifier.3.weight",
    }
    if not crnn_keys.issubset(set(state_dict)):
        return None

    num_classes = int(state_dict["classifier.3.weight"].shape[0])
    if genres and len(genres) == num_classes:
        num_classes = len(genres)
    hidden_size = int(state_dict["gru.weight_hh_l0"].shape[1])
    num_layers = max(
        int(key.split("_l")[1].split("_")[0])
        for key in state_dict
        if key.startswith("gru.weight_ih_l")
    ) + 1
    dropout = float(checkpoint_config.get("crnn_dropout", 0.3)) if checkpoint_config else 0.3
    return CrnnGenreClassifier(
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )


def build_builtin_resnet(
    state_dict: dict[str, torch.Tensor],
    checkpoint_config: dict[str, object] | None,
    genres: Sequence[str] | None,
) -> nn.Module | None:
    resnet_keys = {
        "stem.0.weight",
        "stage1.0.conv1.weight",
        "stage4.1.conv2.weight",
        "classifier.2.weight",
    }
    if not resnet_keys.issubset(set(state_dict)):
        return None

    num_classes = int(state_dict["classifier.2.weight"].shape[0])
    if genres and len(genres) == num_classes:
        num_classes = len(genres)
    dropout = float(checkpoint_config.get("cnn_dropout", 0.3)) if checkpoint_config else 0.3
    return SmallResNetGenreClassifier(num_classes=num_classes, dropout=dropout)


def build_builtin_model(
    state_dict: dict[str, torch.Tensor],
    checkpoint_config: dict[str, object] | None,
    genres: Sequence[str] | None,
) -> nn.Module | None:
    keys = set(state_dict)
    specaug_keys = {
        "features.0.block.0.weight",
        "features.2.block.0.weight",
        "features.4.block.0.weight",
        "features.6.block.0.weight",
        "classifier.1.weight",
        "classifier.4.weight",
    }
    if not specaug_keys.issubset(keys):
        return (
            build_builtin_mobilenet_v2(state_dict, genres)
            or build_builtin_ast(state_dict, genres)
            or build_builtin_crnn(state_dict, checkpoint_config, genres)
            or build_builtin_resnet(state_dict, checkpoint_config, genres)
        )

    if checkpoint_config and "num_classes" in checkpoint_config:
        num_classes = int(checkpoint_config["num_classes"])
    elif genres:
        num_classes = len(genres)
    else:
        num_classes = int(state_dict["classifier.4.weight"].shape[0])

    dropout = float(checkpoint_config.get("cnn_dropout", 0.3)) if checkpoint_config else 0.3
    return SpecAugCnn(num_classes=num_classes, dropout=dropout)


def build_builtin_mobilenet_v2(
    state_dict: dict[str, torch.Tensor],
    genres: Sequence[str] | None,
) -> nn.Module | None:
    keys = set(state_dict)
    mobilenet_keys = {
        "features.0.0.weight",
        "features.1.conv.0.0.weight",
        "features.18.0.weight",
        "classifier.1.weight",
    }
    if not mobilenet_keys.issubset(keys):
        return None

    try:
        from torchvision.models import mobilenet_v2
    except ModuleNotFoundError as error:
        raise explain_missing_dependency(error, Path("pond_best.pt")) from error

    num_classes = int(state_dict["classifier.1.weight"].shape[0])
    if genres and len(genres) != num_classes:
        num_classes = len(genres)

    model = mobilenet_v2(weights=None, num_classes=num_classes)
    first_weight = state_dict["features.0.0.weight"]
    in_channels = int(first_weight.shape[1])
    out_channels = int(first_weight.shape[0])
    model.features[0][0] = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False,
    )
    return model


def is_ast_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    ast_keys = {
        "audio_spectrogram_transformer.embeddings.position_embeddings",
        "audio_spectrogram_transformer.embeddings.patch_embeddings.projection.weight",
        "audio_spectrogram_transformer.encoder.layer.0.attention.attention.query.weight",
        "classifier.dense.weight",
    }
    return ast_keys.issubset(set(state_dict))


def build_builtin_ast(
    state_dict: dict[str, torch.Tensor],
    genres: Sequence[str] | None,
) -> nn.Module | None:
    if not is_ast_state_dict(state_dict):
        return None

    try:
        from transformers import ASTConfig, ASTForAudioClassification
    except ModuleNotFoundError as error:
        raise explain_missing_dependency(error, Path("ast_fma_best.pt")) from error

    classifier_weight = state_dict["classifier.dense.weight"]
    patch_weight = state_dict["audio_spectrogram_transformer.embeddings.patch_embeddings.projection.weight"]
    position_embeddings = state_dict["audio_spectrogram_transformer.embeddings.position_embeddings"]

    num_labels = int(classifier_weight.shape[0])
    hidden_size = int(classifier_weight.shape[1])
    patch_size = int(patch_weight.shape[-1])
    num_positions = int(position_embeddings.shape[1])

    num_mel_bins = 128
    time_stride = 10
    frequency_stride = 10
    num_frequency_patches = (num_mel_bins - patch_size) // frequency_stride + 1
    num_time_patches = (num_positions - 2) // num_frequency_patches
    max_length = (num_time_patches - 1) * time_stride + patch_size

    config = ASTConfig(
        num_labels=num_labels if not genres else len(genres),
        hidden_size=hidden_size,
        num_hidden_layers=max(
            int(key.split(".")[3])
            for key in state_dict
            if key.startswith("audio_spectrogram_transformer.encoder.layer.")
        )
        + 1,
        num_attention_heads=12,
        intermediate_size=int(state_dict["audio_spectrogram_transformer.encoder.layer.0.intermediate.dense.weight"].shape[0]),
        patch_size=patch_size,
        frequency_stride=frequency_stride,
        time_stride=time_stride,
        num_mel_bins=num_mel_bins,
        max_length=max_length,
    )
    return ASTForAudioClassification(config)


class Backend:
    name: str
    class_names: list[str] | None
    config: dict[str, object] | None

    def predict(self, model_input: torch.Tensor) -> np.ndarray:
        raise NotImplementedError


def infer_model_input_channels(model: nn.Module) -> int | None:
    conv_stem = getattr(model, "conv_stem", None)
    if isinstance(conv_stem, nn.Conv2d):
        return int(conv_stem.in_channels)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return int(module.in_channels)

    return None


def adapt_input_channels(model_input: torch.Tensor, expected_channels: int | None) -> torch.Tensor:
    if expected_channels is None or model_input.ndim != 4:
        return model_input

    actual_channels = int(model_input.shape[1])
    if actual_channels == expected_channels:
        return model_input

    if actual_channels == 1 and expected_channels == 3:
        return model_input.repeat(1, 3, 1, 1)

    if actual_channels == 3 and expected_channels == 1:
        return model_input.mean(dim=1, keepdim=True)

    raise ValueError(
        f"Model expects {expected_channels} input channels but preprocessor produced {actual_channels}."
    )


def adapt_ast_input(model_input: torch.Tensor, max_length: int = 1024, num_mel_bins: int = 128) -> torch.Tensor:
    if model_input.ndim == 4:
        if model_input.shape[1] != 1:
            model_input = model_input.mean(dim=1, keepdim=True)
        model_input = model_input.squeeze(1).transpose(1, 2)
    elif model_input.ndim == 3:
        if model_input.shape[1] == num_mel_bins:
            model_input = model_input.transpose(1, 2)
    else:
        raise ValueError(f"AST expects a 3D/4D spectrogram input, got shape {tuple(model_input.shape)}")

    if model_input.shape[-1] != num_mel_bins:
        raise ValueError(
            f"AST expects {num_mel_bins} mel bins, got input shape {tuple(model_input.shape)}"
        )

    current_length = int(model_input.shape[1])
    if current_length < max_length:
        model_input = F.pad(model_input, (0, 0, 0, max_length - current_length))
    elif current_length > max_length:
        model_input = model_input[:, -max_length:, :]

    return model_input


@dataclass
class OnnxBackend(Backend):
    path: Path
    session: object
    input_name: str
    class_names: list[str] | None = None
    config: dict[str, object] | None = None

    @property
    def name(self) -> str:
        return self.path.name

    def predict(self, model_input: torch.Tensor) -> np.ndarray:
        inputs = model_input.detach().cpu().numpy().astype(np.float32)
        outputs = self.session.run(None, {self.input_name: inputs})
        return as_probs(outputs[0])


@dataclass
class TorchBackend(Backend):
    path: Path
    model: nn.Module
    device: torch.device
    class_names: list[str] | None = None
    config: dict[str, object] | None = None
    input_channels: int | None = None
    input_adapter: str = "auto"
    ast_max_length: int = 1024
    ast_num_mel_bins: int = 128

    @property
    def name(self) -> str:
        return self.path.name

    def predict(self, model_input: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            if self.input_adapter == "ast":
                model_input = adapt_ast_input(model_input, self.ast_max_length, self.ast_num_mel_bins)
            else:
                model_input = adapt_input_channels(model_input, self.input_channels)
            output = self.model(model_input.to(self.device))
            if hasattr(output, "logits"):
                output = output.logits
            elif isinstance(output, dict) and "logits" in output:
                output = output["logits"]
            if isinstance(output, (tuple, list)):
                output = output[0]
            return as_probs(output)


class EnsemblePredictor:
    def __init__(self, backends: Sequence[Backend]) -> None:
        if not backends:
            raise ValueError("At least one model is required")
        self.backends = list(backends)

    @property
    def names(self) -> list[str]:
        return [backend.name for backend in self.backends]

    @property
    def class_names(self) -> list[str] | None:
        for backend in self.backends:
            if backend.class_names:
                return backend.class_names
        return None

    @property
    def config(self) -> dict[str, object] | None:
        for backend in self.backends:
            if backend.config:
                return backend.config
        return None

    def predict(self, model_input: torch.Tensor) -> np.ndarray:
        model_probs = [backend.predict(model_input) for backend in self.backends]
        sizes = {prob.shape[0] for prob in model_probs}
        if len(sizes) != 1:
            details = ", ".join(f"{name}:{prob.shape[0]}" for name, prob in zip(self.names, model_probs))
            raise ValueError(f"Models returned different class counts ({details})")
        return np.mean(model_probs, axis=0).astype(np.float32)


def load_onnx_backend(path: Path) -> OnnxBackend:
    if ort is None:
        raise ImportError("onnxruntime is not installed. Install it or use a .pt model.")

    providers = ["CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available:
        providers.insert(0, "CUDAExecutionProvider")

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 4
    session = ort.InferenceSession(str(path), sess_options=session_options, providers=providers)
    input_name = session.get_inputs()[0].name
    return OnnxBackend(path=path, session=session, input_name=input_name)


def load_factory(factory_spec: str | None) -> Callable[..., nn.Module] | None:
    if not factory_spec:
        return None

    module_ref, function_name = factory_spec.rsplit(":", 1)
    module_path = Path(module_ref)

    if module_path.suffix == ".py" or module_path.exists():
        module_path = module_path.resolve()
        sys.path.insert(0, str(module_path.parent))
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import factory module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_path.stem] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)

    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"Factory is not callable: {factory_spec}")
    return factory


def build_model_from_factory(
    factory: Callable[..., nn.Module],
    genres: Sequence[str] | None,
) -> nn.Module:
    attempts: list[tuple[tuple[object, ...], dict[str, object]]] = []
    if genres:
        attempts.extend(
            [
                ((), {"num_classes": len(genres)}),
                ((len(genres),), {}),
            ]
        )
    attempts.append(((), {}))

    last_error: Exception | None = None
    for args, kwargs in attempts:
        try:
            model = factory(*args, **kwargs)
            if not isinstance(model, nn.Module):
                raise TypeError("Factory must return torch.nn.Module")
            return model
        except TypeError as error:
            last_error = error

    raise TypeError(f"Could not call model factory. Last error: {last_error}")


def is_state_dict(value: object) -> bool:
    return isinstance(value, dict) and bool(value) and all(torch.is_tensor(v) for v in value.values())


def find_state_dict(checkpoint: object) -> dict[str, torch.Tensor] | None:
    if is_state_dict(checkpoint):
        return checkpoint

    if not isinstance(checkpoint, dict):
        return None

    for key in ("state_dict", "model_state_dict", "net_state_dict"):
        value = checkpoint.get(key)
        if is_state_dict(value):
            return value

    for key in ("model", "net"):
        value = checkpoint.get(key)
        if is_state_dict(value):
            return value

    return None


def strip_common_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("module.", "model.", "net.")
    stripped = state_dict
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in stripped):
                stripped = {key[len(prefix) :]: value for key, value in stripped.items()}
                changed = True
                break
    return stripped


def torch_load(path: Path) -> object:
    numpy_multiarray = importlib.import_module("numpy.core.multiarray")
    safe_types = [
        (numpy_multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        type(np.dtype("float64")),
        type(np.dtype("int64")),
    ]
    try:
        with safe_globals(safe_types):
            return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def explain_missing_dependency(error: ModuleNotFoundError, model_path: Path) -> ImportError:
    package = error.name or str(error)
    hint = OPTIONAL_MODEL_DEPENDENCIES.get(
        package,
        f"Install the missing package with: pip install {package}",
    )
    return ImportError(
        f"Cannot load model '{model_path.name}' because Python package '{package}' is not installed. "
        f"{hint}. If you want the model that works without extra EfficientNet dependencies, choose "
        "models\\cnn_specaug_best.pt."
    )


def load_torch_backend(
    path: Path,
    device: torch.device,
    factory: Callable[..., nn.Module] | None,
    genres: Sequence[str] | None,
) -> TorchBackend:
    try:
        scripted = torch.jit.load(str(path), map_location=device)
        scripted.eval()
        return TorchBackend(
            path=path,
            model=scripted,
            device=device,
            input_channels=infer_model_input_channels(scripted),
        )
    except Exception:
        pass

    try:
        checkpoint = torch_load(path)
    except ModuleNotFoundError as error:
        raise explain_missing_dependency(error, path) from error
    class_names = extract_class_names(checkpoint)
    checkpoint_config = extract_config(checkpoint)
    input_adapter = "auto"

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state_dict = find_state_dict(checkpoint)
        if state_dict is None:
            raise ValueError(
                f"{path} is not TorchScript, an nn.Module, or a recognized state_dict checkpoint"
            )
        if factory is None:
            model = build_builtin_model(state_dict, checkpoint_config, class_names or genres)
            if model is None:
                raise ValueError(
                    f"{path} contains weights only. Pass --model-factory module_or_file:create_model "
                    "so the pipeline can rebuild the architecture before loading the weights."
                )
        else:
            model = build_model_from_factory(factory, class_names or genres)
        if is_ast_state_dict(state_dict):
            input_adapter = "ast"
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            model.load_state_dict(strip_common_prefix(state_dict))

    model.to(device)
    model.eval()
    return TorchBackend(
        path=path,
        model=model,
        device=device,
        class_names=class_names,
        config=checkpoint_config,
        input_channels=infer_model_input_channels(model),
        input_adapter=input_adapter,
    )


def load_backends(
    model_paths: Sequence[Path],
    factories: Sequence[str],
    device: torch.device,
    genres: Sequence[str] | None,
) -> list[Backend]:
    backends: list[Backend] = []
    loaded_factories: dict[str, Callable[..., nn.Module] | None] = {}

    for index, path in enumerate(model_paths):
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".onnx":
            backends.append(load_onnx_backend(path))
            continue

        if suffix not in (".pt", ".pth", ".ckpt"):
            raise ValueError(f"Unsupported model extension: {path}")

        factory_spec = None
        if len(factories) == 1:
            factory_spec = factories[0]
        elif len(factories) == len(model_paths):
            factory_spec = factories[index]
        elif factories:
            raise ValueError("Pass exactly one --model-factory or one factory per --model")

        if factory_spec not in loaded_factories:
            loaded_factories[factory_spec or ""] = load_factory(factory_spec)
        backends.append(load_torch_backend(path, device, loaded_factories[factory_spec or ""], genres))

    return backends


def load_audio_file(path: str | Path, sample_rate: int) -> np.ndarray:
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if sf is not None:
        try:
            data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
            waveform = torch.from_numpy(data.T).float()
        except Exception:
            waveform = None
            sr = None
    else:
        waveform = None
        sr = None

    if waveform is None:
        import torchaudio

        waveform, sr = torchaudio.load(str(audio_path))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        waveform = T.Resample(sr, sample_rate)(waveform)

    return waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)


def iter_file_windows(
    audio: np.ndarray,
    window_samples: int,
    hop_samples: int,
    include_tail: bool,
) -> Iterable[tuple[int, int, np.ndarray]]:
    total_samples = len(audio)
    if total_samples < window_samples:
        padded = np.pad(audio, (0, window_samples - total_samples))
        yield 1, total_samples, padded
        return

    window_index = 0
    end_positions = list(range(window_samples, total_samples + 1, hop_samples))
    if include_tail and end_positions and end_positions[-1] != total_samples:
        end_positions.append(total_samples)

    for end in end_positions:
        window_index += 1
        yield window_index, end, audio[end - window_samples : end]


def majority_vote(indices: Sequence[int], tie_scores: np.ndarray | None = None) -> int:
    if not indices:
        raise ValueError("Cannot vote over an empty prediction list")

    counts = Counter(indices)
    max_votes = max(counts.values())
    candidates = [idx for idx, votes in counts.items() if votes == max_votes]
    if len(candidates) == 1 or tie_scores is None:
        return candidates[0]

    return max(candidates, key=lambda idx: float(tie_scores[idx]))


def display_results(
    genres: Sequence[str],
    soft_probs: np.ndarray,
    latest_probs: np.ndarray,
    window_index: int,
    elapsed_sec: float,
    window_sec: float,
    hop_sec: float,
    vote_windows: int,
    inference_sec: float,
    end_sec: float | None = None,
) -> None:
    clear_screen()
    soft_idx = int(np.argmax(soft_probs))
    latest_idx = int(np.argmax(latest_probs))
    rtf = inference_sec / hop_sec

    print("=" * 72)
    print("Real-time Music Genre Classification")
    print("=" * 72)
    print(
        f"Window {window_index} | elapsed {elapsed_sec:6.1f}s | "
        f"model window {window_sec:g}s | hop {hop_sec:g}s | soft vote last {vote_windows}"
    )
    if end_sec is not None:
        print(f"Audio position: {end_sec:6.1f}s")
    print(f"Inference: {inference_sec * 1000:7.1f} ms | RTF vs hop: {rtf:.3f}")
    print("-" * 72)

    order = np.argsort(soft_probs)[::-1]
    for idx in order:
        marker = "<" if idx == soft_idx else " "
        print(f"{genres[idx]:<18} {format_bar(float(soft_probs[idx]))} {soft_probs[idx] * 100:6.2f}% {marker}")

    print("-" * 72)
    print(f"Current 3-window soft vote : {genres[soft_idx]} ({soft_probs[soft_idx] * 100:.2f}%)")
    print(f"Latest single window       : {genres[latest_idx]} ({latest_probs[latest_idx] * 100:.2f}%)")
    print("Press Ctrl+C to stop and print final majority vote.")


def print_final_summary(
    genres: Sequence[str],
    latest_history: Sequence[np.ndarray],
    soft_history: Sequence[np.ndarray],
    soft_decisions: Sequence[int],
    inference_times: Sequence[float],
    window_sec: float,
    hop_sec: float,
    true_genre: str | None = None,
) -> None:
    if not latest_history:
        print("[WARN] No complete 15-second window was processed.")
        return

    latest_mean = np.mean(latest_history, axis=0)
    soft_mean = np.mean(soft_history, axis=0)
    final_idx = majority_vote(soft_decisions, soft_mean)
    raw_majority_idx = majority_vote([int(np.argmax(probs)) for probs in latest_history], latest_mean)
    avg_inference = float(np.mean(inference_times)) if inference_times else 0.0

    print()
    print("=" * 72)
    print("Final Result")
    print("=" * 72)
    print(f"Windows processed          : {len(latest_history)}")
    print(f"Model window / hop         : {window_sec:g}s / {hop_sec:g}s")
    print(f"Avg inference              : {avg_inference * 1000:.1f} ms")
    print(f"Avg RTF vs hop             : {avg_inference / hop_sec:.3f}")
    print("-" * 72)
    print(f"Final genre                : {genres[final_idx]}")
    print(f"Majority over soft votes   : {genres[final_idx]}")
    print(f"Majority over raw windows  : {genres[raw_majority_idx]}")
    print(f"All-window soft vote       : {genres[int(np.argmax(latest_mean))]}")

    if true_genre:
        matched = genres[final_idx].lower() == true_genre.lower()
        print(f"True genre                 : {true_genre}")
        print(f"Match                      : {'yes' if matched else 'no'}")

    print("-" * 72)
    print("Top probabilities from all-window average:")
    for idx in np.argsort(latest_mean)[::-1][: min(5, len(genres))]:
        print(f"{genres[idx]:<18} {latest_mean[idx] * 100:6.2f}%")
    print("=" * 72)


def process_window(
    predictor: EnsemblePredictor,
    preprocessor: AudioPreprocessor,
    window: np.ndarray,
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    model_input = preprocessor(window)
    probs = predictor.predict(model_input)
    elapsed = time.perf_counter() - t0
    return probs, elapsed


def run_from_file(
    predictor: EnsemblePredictor,
    preprocessor: AudioPreprocessor,
    genres: Sequence[str],
    file_path: str | Path,
    hop_samples: int,
    vote_windows: int,
    include_tail: bool,
    simulate_realtime: bool,
    true_genre: str | None,
) -> None:
    audio = load_audio_file(file_path, preprocessor.sample_rate)
    duration = len(audio) / preprocessor.sample_rate
    print(f"[INFO] Loaded audio: {file_path}")
    print(f"[INFO] Duration: {duration:.2f}s")

    latest_history: list[np.ndarray] = []
    soft_history: list[np.ndarray] = []
    soft_decisions: list[int] = []
    inference_times: list[float] = []
    vote_buffer: deque[np.ndarray] = deque(maxlen=vote_windows)
    start_time = time.time()

    try:
        for window_index, end_sample, window in iter_file_windows(
            audio,
            preprocessor.window_samples,
            hop_samples,
            include_tail=include_tail,
        ):
            loop_start = time.perf_counter()
            probs, inference_sec = process_window(predictor, preprocessor, window)

            latest_history.append(probs)
            inference_times.append(inference_sec)
            vote_buffer.append(probs)

            soft_probs = np.mean(vote_buffer, axis=0)
            soft_history.append(soft_probs)
            soft_decisions.append(int(np.argmax(soft_probs)))

            display_results(
                genres=genres,
                soft_probs=soft_probs,
                latest_probs=probs,
                window_index=window_index,
                elapsed_sec=time.time() - start_time,
                window_sec=preprocessor.window_sec,
                hop_sec=hop_samples / preprocessor.sample_rate,
                vote_windows=min(vote_windows, len(vote_buffer)),
                inference_sec=inference_sec,
                end_sec=end_sample / preprocessor.sample_rate,
            )

            if simulate_realtime:
                sleep_time = (hop_samples / preprocessor.sample_rate) - (time.perf_counter() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

    print_final_summary(
        genres=genres,
        latest_history=latest_history,
        soft_history=soft_history,
        soft_decisions=soft_decisions,
        inference_times=inference_times,
        window_sec=preprocessor.window_sec,
        hop_sec=hop_samples / preprocessor.sample_rate,
        true_genre=true_genre,
    )


def run_realtime(
    predictor: EnsemblePredictor,
    preprocessor: AudioPreprocessor,
    genres: Sequence[str],
    hop_samples: int,
    vote_windows: int,
    mic_device: int | str | None = None,
) -> None:
    if sd is None:
        raise ImportError("sounddevice is not installed. Install it or run with --file.")

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def audio_callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"[WARN] {status}", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    latest_history: list[np.ndarray] = []
    soft_history: list[np.ndarray] = []
    soft_decisions: list[int] = []
    inference_times: list[float] = []
    vote_buffer: deque[np.ndarray] = deque(maxlen=vote_windows)

    buffer = np.empty(0, dtype=np.float32)
    samples_since_prediction = 0
    start_time = time.time()

    print("[INFO] Starting microphone stream")
    print(
        f"[INFO] Need {preprocessor.window_sec:g}s before first prediction; "
        f"then update every {hop_samples / preprocessor.sample_rate:g}s."
    )

    try:
        with sd.InputStream(
            samplerate=preprocessor.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=1024,
            device=mic_device,
            callback=audio_callback,
        ):
            while True:
                try:
                    data = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                buffer = np.concatenate([buffer, data.astype(np.float32, copy=False)])
                if buffer.size > preprocessor.window_samples:
                    buffer = buffer[-preprocessor.window_samples :]

                samples_since_prediction += data.size

                if buffer.size < preprocessor.window_samples or samples_since_prediction < hop_samples:
                    continue

                samples_since_prediction = 0
                probs, inference_sec = process_window(predictor, preprocessor, buffer)

                latest_history.append(probs)
                inference_times.append(inference_sec)
                vote_buffer.append(probs)

                soft_probs = np.mean(vote_buffer, axis=0)
                soft_history.append(soft_probs)
                soft_decisions.append(int(np.argmax(soft_probs)))

                display_results(
                    genres=genres,
                    soft_probs=soft_probs,
                    latest_probs=probs,
                    window_index=len(latest_history),
                    elapsed_sec=time.time() - start_time,
                    window_sec=preprocessor.window_sec,
                    hop_sec=hop_samples / preprocessor.sample_rate,
                    vote_windows=min(vote_windows, len(vote_buffer)),
                    inference_sec=inference_sec,
                )

    except KeyboardInterrupt:
        print("\n[INFO] Stopped microphone stream.")

    print_final_summary(
        genres=genres,
        latest_history=latest_history,
        soft_history=soft_history,
        soft_decisions=soft_decisions,
        inference_times=inference_times,
        window_sec=preprocessor.window_sec,
        hop_sec=hop_samples / preprocessor.sample_rate,
    )


def expand_models(model_args: Sequence[str] | None) -> list[Path]:
    if not model_args:
        return [resolve_default_model()]

    paths: list[Path] = []
    for raw in model_args:
        paths.extend(Path(item.strip()) for item in raw.split(",") if item.strip())
    return paths


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def find_files(folder: Path, extensions: Sequence[str]) -> list[Path]:
    if not folder.exists():
        return []

    extensions = tuple(ext.lower() for ext in extensions)
    files = [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in extensions]

    def sort_key(path: Path) -> tuple[int, str]:
        suffix_priority = {".pt": 0, ".pth": 1, ".ckpt": 2, ".onnx": 3}
        return suffix_priority.get(path.suffix.lower(), 10), path.name.lower()

    return sorted(files, key=sort_key)


def display_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def prompt_choice(title: str, items: Sequence[Path], base: Path) -> Path:
    while True:
        print()
        print(title)
        print("-" * len(title))
        for index, item in enumerate(items, start=1):
            print(f"{index:2d}. {display_path(item, base)}")

        raw = input("Select number: ").strip()
        if not raw.isdigit():
            print("Please type a number from the list.")
            continue

        selected = int(raw)
        if 1 <= selected <= len(items):
            return items[selected - 1]

        print("That number is not in the list.")


def prompt_mode() -> str:
    while True:
        print()
        print("Input source")
        print("------------")
        print(" 1. Music file from songs/")
        print(" 2. Microphone")
        raw = input("Select number: ").strip()
        if raw == "1":
            return "file"
        if raw == "2":
            return "mic"
        print("Please choose 1 or 2.")


def apply_interactive_selection(args: argparse.Namespace) -> None:
    root = script_dir()
    models_dir = Path(args.models_dir) if args.models_dir else root / "models"
    songs_dir = Path(args.songs_dir) if args.songs_dir else root / "songs"

    print("=" * 72)
    print("Realtime Music Genre Classifier")
    print("=" * 72)
    print(f"Models folder : {models_dir}")
    print(f"Songs folder  : {songs_dir}")

    if not args.model:
        model_files = find_files(models_dir, MODEL_EXTENSIONS)
        if not model_files:
            raise FileNotFoundError(f"No model files found in {models_dir}")
        args.model = [str(prompt_choice("Choose model", model_files, models_dir))]

    mode = args.mode or ("file" if args.file else None) or prompt_mode()
    if mode == "mic":
        args.file = None
        return

    if mode != "file":
        raise ValueError("--mode must be file or mic")

    if not args.file:
        song_files = find_files(songs_dir, AUDIO_EXTENSIONS)
        if not song_files:
            raise FileNotFoundError(
                f"No audio files found in {songs_dir}. Put .mp3/.wav files there and run python realtime.py again."
            )
        args.file = str(prompt_choice("Choose song", song_files, songs_dir))

    if not args.simulate_realtime:
        args.simulate_realtime = True


def validate_class_count(genres: Sequence[str], predictor: EnsemblePredictor, preprocessor: AudioPreprocessor) -> None:
    silence = np.zeros(preprocessor.window_samples, dtype=np.float32)
    try:
        probs = predictor.predict(preprocessor(silence))
    except Exception as error:
        raise RuntimeError(
            "The model could not run on the configured input shape. "
            "Check --window-sec, --input-format, mel settings, and whether the checkpoint was trained/exported "
            f"for {preprocessor.window_sec:g}s audio windows."
        ) from error
    if probs.shape[0] != len(genres):
        raise ValueError(
            f"Model returns {probs.shape[0]} classes but genre list has {len(genres)} labels. "
            "Pass --genres or --label-map with the training label order."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sliding-window real-time music genre classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Model path. Repeat for model ensemble, or pass comma-separated paths.",
    )
    parser.add_argument(
        "--model-factory",
        action="append",
        default=[],
        help=(
            "Factory for .pt/.pth state_dict checkpoints, e.g. src.models:create_model "
            "or C:\\path\\model_def.py:create_model. Repeat per --model if architectures differ."
        ),
    )
    parser.add_argument("--file", help="Audio file path. If omitted, microphone mode is used.")
    parser.add_argument("--mode", choices=("file", "mic"), help="Interactive input source shortcut.")
    parser.add_argument("--genre", help="True genre label for file-mode comparison.")
    parser.add_argument("--genres", help="Comma-separated genre labels in the exact training order.")
    parser.add_argument("--label-map", help="CSV with label,genre columns in training label order.")
    parser.add_argument("--models-dir", help="Folder used by interactive mode for model selection.")
    parser.add_argument("--songs-dir", help="Folder used by interactive mode for song selection.")
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--window-sec", type=float, default=None)
    parser.add_argument("--hop-sec", type=float, default=HOP_SEC)
    parser.add_argument("--vote-windows", type=int, default=VOTE_WINDOWS)
    parser.add_argument("--n-mels", type=int, default=None)
    parser.add_argument("--n-fft", type=int, default=None)
    parser.add_argument("--mel-hop-length", type=int, default=None)
    parser.add_argument("--f-min", type=float, default=None)
    parser.add_argument("--f-max", type=float, default=None)
    parser.add_argument("--input-format", choices=("mel4d", "mel3d", "waveform"), default="mel4d")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-tail", action="store_true", help="Do not add the final non-aligned file window.")
    parser.add_argument("--simulate-realtime", action="store_true", help="Sleep one hop between file windows.")
    parser.add_argument("--interactive", action="store_true", help="Open a simple menu to choose model and input source.")
    parser.add_argument("--list-devices", action="store_true", help="List sounddevice input devices and exit.")
    parser.add_argument("--mic-device", help="sounddevice input device index or name for microphone mode.")
    parser.add_argument("--debug", action="store_true", help="Show full Python tracebacks for errors.")
    args = parser.parse_args()

    if args.list_devices:
        if sd is None:
            raise ImportError("sounddevice is not installed.")
        print(sd.query_devices())
        return

    if args.interactive or len(sys.argv) == 1 or args.mode is not None or (args.file and not args.model):
        apply_interactive_selection(args)

    model_paths = expand_models(args.model)
    label_map_genres = read_label_map(args.label_map)
    cli_genres = parse_genres(args.genres)
    genres = cli_genres or label_map_genres

    device = choose_device(args.device)
    backends = load_backends(model_paths, args.model_factory, device, genres)
    predictor = EnsemblePredictor(backends)
    checkpoint_config = predictor.config or {}

    genres = genres or predictor.class_names or DEFAULT_GENRES

    sample_rate = int(args.sample_rate or checkpoint_config.get("sample_rate", SAMPLE_RATE))
    window_sec = float(args.window_sec or checkpoint_config.get("chunk_duration", WINDOW_SEC))
    n_mels = int(args.n_mels or checkpoint_config.get("n_mels", N_MELS))
    n_fft = int(args.n_fft or checkpoint_config.get("n_fft", N_FFT))
    mel_hop_length = int(args.mel_hop_length or checkpoint_config.get("hop_length", MEL_HOP_LENGTH))
    f_min = float(args.f_min if args.f_min is not None else checkpoint_config.get("fmin", F_MIN))
    raw_f_max = args.f_max if args.f_max is not None else checkpoint_config.get("fmax", F_MAX)
    f_max = None if raw_f_max is None else float(raw_f_max)

    preprocessor = AudioPreprocessor(
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
    if args.vote_windows <= 0:
        raise ValueError("--vote-windows must be positive")

    print(f"[INFO] Models: {', '.join(predictor.names)}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Labels: {', '.join(genres)}")
    print(
        f"[INFO] Audio config: sr={sample_rate}, window={window_sec:g}s, "
        f"hop={args.hop_sec:g}s, n_mels={n_mels}, n_fft={n_fft}, "
        f"mel_hop={mel_hop_length}, f_min={f_min:g}, f_max={f_max}"
    )
    print(f"[INFO] Input format: {args.input_format}")
    validate_class_count(genres, predictor, preprocessor)

    if args.file:
        run_from_file(
            predictor=predictor,
            preprocessor=preprocessor,
            genres=genres,
            file_path=args.file,
            hop_samples=hop_samples,
            vote_windows=args.vote_windows,
            include_tail=not args.no_tail,
            simulate_realtime=args.simulate_realtime,
            true_genre=args.genre,
        )
    else:
        run_realtime(
            predictor=predictor,
            preprocessor=preprocessor,
            genres=genres,
            hop_samples=hop_samples,
            vote_windows=args.vote_windows,
            mic_device=parse_mic_device(args.mic_device),
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--debug" in sys.argv:
            raise
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
