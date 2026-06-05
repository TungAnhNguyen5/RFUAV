from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


SUPPORTED_EXTS = {".dat", ".bin", ".iq", ".npy"}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_output_dir(requested: str) -> Path:
    """
    Always place relative experiment outputs inside output_iq/.
    Example:
        --output-dir stft_balanced_512
        -> output_iq/stft_balanced_512
    """
    output_root = Path("output_iq")
    requested_path = Path(requested)

    if requested_path.is_absolute():
        output_dir = requested_path
    elif requested_path.parts and requested_path.parts[0] == output_root.name:
        output_dir = requested_path
    else:
        output_dir = output_root / requested_path

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def count_complex_samples(path: Path) -> int:
    """
    Count usable complex IQ samples.
    For raw .iq/.dat/.bin files, this assumes float32 interleaved IQ:
    I0, Q0, I1, Q1, ...
    Incomplete trailing bytes are ignored.
    """
    if path.suffix.lower() == ".npy":
        arr = np.load(path, mmap_mode="r")
        if np.iscomplexobj(arr):
            return arr.shape[0]
        return arr.shape[0] // 2

    size_bytes = path.stat().st_size
    itemsize = np.dtype(np.float32).itemsize

    usable_float32_values = size_bytes // itemsize
    usable_float32_values -= usable_float32_values % 2

    dropped_bytes = size_bytes - usable_float32_values * itemsize
    if dropped_bytes > 0:
        print(f"Warning: ignoring {dropped_bytes} trailing bytes in {path}")

    return usable_float32_values // 2


def read_iq_chunk(path: Path, start: int, chunk_size: int) -> np.ndarray:
    """
    Read one complex IQ chunk.
    Raw files are assumed to be float32 interleaved IQ:
    I0, Q0, I1, Q1, ...
    """
    if path.suffix.lower() == ".npy":
        arr = np.load(path, mmap_mode="r")

        if np.iscomplexobj(arr):
            iq = arr[start:start + chunk_size].astype(np.complex64)
        else:
            raw = arr[2 * start: 2 * (start + chunk_size)].astype(np.float32)
            iq = raw[0::2] + 1j * raw[1::2]

        return iq.astype(np.complex64)

    size_bytes = path.stat().st_size
    itemsize = np.dtype(np.float32).itemsize

    usable_float32_values = size_bytes // itemsize
    usable_float32_values -= usable_float32_values % 2

    raw = np.memmap(
        path,
        dtype=np.float32,
        mode="r",
        shape=(usable_float32_values,),
    )

    raw_chunk = raw[2 * start: 2 * (start + chunk_size)]
    iq = raw_chunk[0::2] + 1j * raw_chunk[1::2]

    return iq.astype(np.complex64)


def iq_to_stft_feature(
    iq: np.ndarray,
    nfft: int,
    hop: int,
    max_time_frames: int = 0,
) -> np.ndarray:
    """
    Convert one IQ chunk into a log-magnitude STFT/spectrogram feature.

    Output shape:
        [frequency_bins, time_frames]
    """
    if iq.size < nfft:
        raise ValueError(f"IQ chunk is shorter than nfft: {iq.size} < {nfft}")

    iq = iq.astype(np.complex64)

    # Remove DC offset
    iq = iq - np.mean(iq)

    # Power normalize
    power = np.mean(np.abs(iq) ** 2)
    iq = iq / np.sqrt(power + 1e-8)

    num_frames = 1 + (len(iq) - nfft) // hop

    # Vectorized framing without copying the full signal repeatedly
    frames = np.lib.stride_tricks.as_strided(
        iq,
        shape=(num_frames, nfft),
        strides=(hop * iq.strides[0], iq.strides[0]),
        writeable=False,
    )

    window = np.hanning(nfft).astype(np.float32)
    frames = frames * window[None, :]

    spectrum = np.fft.fftshift(np.fft.fft(frames, n=nfft, axis=1), axes=1)
    spec = np.log1p(np.abs(spectrum)).astype(np.float32)

    # Shape: [freq, time]
    spec = spec.T

    # Optional time cropping to keep tensors manageable
    if max_time_frames > 0 and spec.shape[1] > max_time_frames:
        spec = spec[:, :max_time_frames]

    # Per-sample standardization
    spec = (spec - spec.mean()) / (spec.std() + 1e-8)

    return spec.astype(np.float32)


def cache_key(path: Path, start: int, chunk_size: int, nfft: int, hop: int, max_time_frames: int) -> str:
    text = f"{path.resolve()}|{start}|{chunk_size}|{nfft}|{hop}|{max_time_frames}"
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class IQSTFTDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        labels: list[int],
        chunk_size: int,
        nfft: int,
        hop: int,
        max_chunks_per_file: int,
        max_time_frames: int = 0,
        cache_dir: Path | None = None,
    ):
        self.files = files
        self.labels = labels
        self.chunk_size = chunk_size
        self.nfft = nfft
        self.hop = hop
        self.max_time_frames = max_time_frames
        self.cache_dir = cache_dir

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

        self.samples: list[tuple[Path, int, int]] = []

        for path, label in zip(files, labels):
            total_complex = count_complex_samples(path)
            num_chunks = total_complex // chunk_size

            if num_chunks <= 0:
                print(f"Skipping short file: {path}")
                continue

            starts = [i * chunk_size for i in range(num_chunks)]

            # Sample evenly across the whole file instead of only taking the beginning
            if max_chunks_per_file > 0 and len(starts) > max_chunks_per_file:
                idxs = np.linspace(0, len(starts) - 1, max_chunks_per_file, dtype=int)
                starts = [starts[i] for i in idxs]

            for start in starts:
                self.samples.append((path, start, label))

        if len(self.samples) == 0:
            raise RuntimeError("No usable IQ chunks found. Check file format and chunk size.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, start, label = self.samples[idx]

        feature = None

        if self.cache_dir is not None:
            key = cache_key(path, start, self.chunk_size, self.nfft, self.hop, self.max_time_frames)
            cache_path = self.cache_dir / f"{key}.npy"

            if cache_path.exists():
                feature = np.load(cache_path).astype(np.float32)

        if feature is None:
            iq = read_iq_chunk(path, start, self.chunk_size)
            feature = iq_to_stft_feature(
                iq,
                nfft=self.nfft,
                hop=self.hop,
                max_time_frames=self.max_time_frames,
            )

            if self.cache_dir is not None:
                np.save(cache_path, feature)

        # Shape for Conv2D: [channels, freq, time]
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return feature_tensor, label_tensor


class SmallSTFT2DCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def collect_files(data_root: Path, class_names: list[str]):
    files = []
    labels = []

    for label, class_name in enumerate(class_names):
        class_dir = data_root / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        class_files = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ]

        if len(class_files) == 0:
            raise RuntimeError(f"No IQ files found in {class_dir}")

        print(f"{class_name}: {len(class_files)} files")

        files.extend(class_files)
        labels.extend([label] * len(class_files))

    return files, labels


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for x, y in tqdm(loader, desc="Valid", leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * y.size(0)
        correct += (preds == y).sum().item()
        total += y.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    return total_loss / total, correct / total, np.array(all_labels), np.array(all_preds)


def save_confusion_matrix(y_true, y_pred, class_names, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(np.float32) / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0)

    ax.set_title("Normalized STFT-based UAV Classifier Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            percent = cm_norm[i, j] * 100
            count = cm[i, j]
            ax.text(
                j,
                i,
                f"{percent:.1f}%\n({count})",
                ha="center",
                va="center",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def write_experiment_config(args, output_dir: Path, class_names: list[str], train_len: int, valid_len: int):
    config_path = output_dir / "experiment_config.txt"

    with open(config_path, "w", encoding="utf-8") as f:
        f.write("STFT IQ Classifier Experiment Config\n")
        f.write("====================================\n\n")
        f.write(f"classes: {class_names}\n")
        f.write(f"data_root: {args.data_root}\n")
        f.write(f"chunk_size: {args.chunk_size}\n")
        f.write(f"nfft: {args.nfft}\n")
        f.write(f"hop: {args.hop}\n")
        f.write(f"max_time_frames: {args.max_time_frames}\n")
        f.write(f"max_chunks_per_file: {args.max_chunks_per_file}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"valid_ratio: {args.valid_ratio}\n")
        f.write(f"train_chunks: {train_len}\n")
        f.write(f"valid_chunks: {valid_len}\n")
        f.write(f"cache_dir: {args.cache_dir}\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["YUNZHUO-H12", "YUNZHUO-H16", "YUNZHUO-H30"],
    )

    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--nfft", type=int, default=512)
    parser.add_argument("--hop", type=int, default=128)
    parser.add_argument("--max-time-frames", type=int, default=0)

    parser.add_argument("--max-chunks-per-file", type=int, default=300)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--valid-ratio", type=float, default=0.2)

    parser.add_argument("--output-dir", type=str, default="stft_baseline")
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    if args.chunk_size < args.nfft:
        raise ValueError("--chunk-size must be >= --nfft")

    set_seed(args.seed)

    data_root = Path(args.data_root)
    output_dir = safe_output_dir(args.output_dir)

    cache_dir = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        if not cache_dir.is_absolute():
            cache_dir = output_dir / cache_dir

    class_names = args.classes
    print("Classes:", class_names)
    print("Output dir:", output_dir)

    files, labels = collect_files(data_root, class_names)
    print(f"Total IQ files: {len(files)}")

    full_ds = IQSTFTDataset(
        files,
        labels,
        chunk_size=args.chunk_size,
        nfft=args.nfft,
        hop=args.hop,
        max_chunks_per_file=args.max_chunks_per_file,
        max_time_frames=args.max_time_frames,
        cache_dir=cache_dir,
    )

    sample_labels = [label for _, _, label in full_ds.samples]
    indices = list(range(len(full_ds)))

    train_idx, valid_idx = train_test_split(
        indices,
        test_size=args.valid_ratio,
        random_state=args.seed,
        stratify=sample_labels,
    )

    train_ds = Subset(full_ds, train_idx)
    valid_ds = Subset(full_ds, valid_idx)

    print(f"Train chunks: {len(train_ds)}")
    print(f"Valid chunks: {len(valid_ds)}")

    # Show spectrogram tensor shape once
    x0, y0 = full_ds[0]
    print(f"Example input tensor shape: {tuple(x0.shape)}  # [channel, freq, time]")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = SmallSTFT2DCNN(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_valid_acc = 0.0
    best_epoch = -1

    write_experiment_config(args, output_dir, class_names, len(train_ds), len(valid_ds))

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        valid_loss, valid_acc, y_true, y_pred = evaluate(
            model, valid_loader, criterion, device
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | "
            f"Valid Acc: {valid_acc:.4f}"
        )

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch + 1

            torch.save(model.state_dict(), output_dir / "best_stft_2dcnn.pt")

            save_confusion_matrix(
                y_true,
                y_pred,
                class_names,
                output_dir / "confusion_matrix.png",
            )

            with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
                f.write(
                    classification_report(
                        y_true,
                        y_pred,
                        target_names=class_names,
                        zero_division=0,
                    )
                )

    print()
    print("Done.")
    print(f"Best valid accuracy: {best_valid_acc:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Saved model to: {output_dir / 'best_stft_2dcnn.pt'}")
    print(f"Saved confusion matrix to: {output_dir / 'confusion_matrix.png'}")
    print(f"Saved report to: {output_dir / 'classification_report.txt'}")
    print(f"Saved config to: {output_dir / 'experiment_config.txt'}")


if __name__ == "__main__":
    main()
