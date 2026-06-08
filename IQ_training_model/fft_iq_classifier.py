from __future__ import annotations

import yaml
import argparse
import random
import shlex
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm


SUPPORTED_EXTS = {".dat", ".bin", ".iq", ".npy"}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_yaml_config(config_path: str | None) -> dict:
    if config_path is None:
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError("YAML config must be a dictionary at the top level.")

    return config

def count_complex_samples(path: Path) -> int:
    if path.suffix.lower() == ".npy":
        arr = np.load(path, mmap_mode="r")
        if np.iscomplexobj(arr):
            return arr.shape[0]
        return arr.shape[0] // 2

    itemsize = np.dtype(np.float32).itemsize
    size_bytes = path.stat().st_size

    usable_values = size_bytes // itemsize

    # Need I/Q pairs, so raw float count must be even
    usable_values = usable_values - (usable_values % 2)

    return usable_values // 2


def read_iq_chunk(path: Path, start: int, chunk_size: int) -> np.ndarray:
    """
    Returns complex IQ chunk with shape [chunk_size].
    Assumes float32 interleaved format:
    I0, Q0, I1, Q1, ...
    Ignores incomplete trailing bytes.
    """
    if path.suffix.lower() == ".npy":
        arr = np.load(path, mmap_mode="r")

        if np.iscomplexobj(arr):
            iq = arr[start:start + chunk_size].astype(np.complex64)
        else:
            raw = arr[2 * start: 2 * (start + chunk_size)].astype(np.float32)
            iq = raw[0::2] + 1j * raw[1::2]

        return iq.astype(np.complex64)

    itemsize = np.dtype(np.float32).itemsize
    size_bytes = path.stat().st_size

    usable_values = size_bytes // itemsize
    usable_values = usable_values - (usable_values % 2)

    raw = np.memmap(
        path,
        dtype=np.float32,
        mode="r",
        shape=(usable_values,),
    )

    raw_chunk = raw[2 * start: 2 * (start + chunk_size)]

    iq = raw_chunk[0::2] + 1j * raw_chunk[1::2]
    return iq.astype(np.complex64)


def iq_to_fft_feature(iq: np.ndarray, nfft: int) -> np.ndarray:
    """
    Converts one IQ chunk into one FFT log-magnitude feature vector.
    Output shape: [nfft]
    """
    iq = iq.astype(np.complex64)

    # Remove DC offset
    iq = iq - np.mean(iq)

    # Power normalize
    power = np.mean(np.abs(iq) ** 2)
    iq = iq / np.sqrt(power + 1e-8)

    # Window before FFT
    window = np.hanning(len(iq)).astype(np.float32)
    iq = iq * window

    # FFT
    spectrum = np.fft.fftshift(np.fft.fft(iq, n=nfft))

    # Log magnitude
    mag = np.log1p(np.abs(spectrum)).astype(np.float32)

    # Per-sample standardization
    mag = (mag - mag.mean()) / (mag.std() + 1e-8)

    return mag.astype(np.float32)


class IQFFTDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        labels: list[int],
        chunk_size: int,
        nfft: int,
        max_chunks_per_file: int,
    ):
        self.files = files
        self.labels = labels
        self.chunk_size = chunk_size
        self.nfft = nfft
        self.samples: list[tuple[Path, int, int]] = []

        for path, label in zip(files, labels):
            total_complex = count_complex_samples(path)
            num_chunks = total_complex // chunk_size

            if num_chunks <= 0:
                continue

            starts = [i * chunk_size for i in range(num_chunks)]

            # Sample chunks evenly across the whole IQ file instead of only using the beginning
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

        iq = read_iq_chunk(path, start, self.chunk_size)
        feature = iq_to_fft_feature(iq, self.nfft)

        # Shape for Conv1D: [channels, length]
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return feature, label


class SmallFFT1DCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(-1)
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


# def save_confusion_matrix(y_true, y_pred, class_names, save_path: Path):
#     cm = confusion_matrix(y_true, y_pred)

#     fig, ax = plt.subplots(figsize=(7, 6))
#     im = ax.imshow(cm)

#     ax.set_title("FFT-based UAV Classifier Confusion Matrix")
#     ax.set_xlabel("Predicted label")
#     ax.set_ylabel("True label")

#     ax.set_xticks(np.arange(len(class_names)))
#     ax.set_yticks(np.arange(len(class_names)))
#     ax.set_xticklabels(class_names, rotation=45, ha="right")
#     ax.set_yticklabels(class_names)

#     for i in range(len(class_names)):
#         for j in range(len(class_names)):
#             ax.text(j, i, str(cm[i, j]), ha="center", va="center")

#     fig.colorbar(im, ax=ax)
#     fig.tight_layout()
#     fig.savefig(save_path, dpi=200)
#     plt.close(fig)

def save_confusion_matrix(y_true, y_pred, class_names, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)

    # Normalize by true class row
    cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0)

    ax.set_title("Normalized FFT-based UAV Classifier Confusion Matrix")
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

def get_executable_line() -> str:
    return " ".join(shlex.quote(x) for x in [sys.executable] + sys.argv)


def write_experiment_config(
    args,
    output_dir: Path,
    class_names: list[str],
    train_len: int,
    valid_len: int,
    total_files: int,
    best_valid_acc: float | None = None,
    best_epoch: int | None = None,
):
    """
    Write a comparable experiment_config.txt.

    This uses the same layout as the STFT script:
        1. Executable Line
        2. Dataset
        3. Feature Settings
        4. Training Settings
        5. Output
        6. Results
    """
    config_path = output_dir / "experiment_config.txt"

    with open(config_path, "w", encoding="utf-8") as f:
        f.write("FFT IQ Classifier Experiment Config\n")
        f.write("===================================\n\n")

        f.write("Executable Line\n")
        f.write("---------------\n")
        f.write(get_executable_line() + "\n\n")

        f.write("Dataset\n")
        f.write("-------\n")
        f.write(f"classes: {class_names}\n")
        f.write(f"data_root: {args.data_root}\n")
        f.write(f"total_iq_files: {total_files}\n")
        f.write(f"train_chunks: {train_len}\n")
        f.write(f"valid_chunks: {valid_len}\n\n")

        f.write("Feature Settings\n")
        f.write("----------------\n")
        f.write("feature_family: FFT\n")
        f.write("feature_type: log-magnitude FFT\n")
        f.write(f"chunk_size: {args.chunk_size}\n")
        f.write(f"nfft: {args.nfft}\n")
        f.write("hop: N/A\n")
        f.write("max_time_frames: N/A\n")
        f.write("window: Hann\n")
        f.write("normalization: DC removal + power normalization + per-sample standardization\n")
        f.write(f"max_chunks_per_file: {args.max_chunks_per_file}\n")
        f.write("cache_dir: N/A\n\n")

        f.write("Training Settings\n")
        f.write("-----------------\n")
        f.write("model: SmallFFT1DCNN\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"learning_rate: {args.lr}\n")
        f.write(f"valid_ratio: {args.valid_ratio}\n")
        f.write(f"seed: {args.seed}\n")
        f.write("num_workers: 0\n\n")

        f.write("Output\n")
        f.write("------\n")
        f.write(f"output_dir: {output_dir}\n")
        f.write("saved_model: best_fft_1dcnn.pt\n")
        f.write("confusion_matrix: confusion_matrix.png\n")
        f.write("classification_report: classification_report.txt\n")
        f.write("experiment_config: experiment_config.txt\n\n")

        f.write("Results\n")
        f.write("-------\n")
        if best_valid_acc is None:
            f.write("best_valid_accuracy: pending\n")
        else:
            f.write(f"best_valid_accuracy: {best_valid_acc:.4f}\n")

        if best_epoch is None:
            f.write("best_epoch: pending\n")
        else:
            f.write(f"best_epoch: {best_epoch}\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=str, required=None)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["YunZhuo-H12", "YunZhuo-H16", "YunZhuo-H30"],
    )

    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--nfft", type=int, default=4096)
    parser.add_argument("--max-chunks-per-file", type=int, default=100)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--valid-ratio", type=float, default=0.2)

    parser.add_argument("--output-dir", type=str, default="outputs_fft")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    config_args, remaining_args = parser.parse_known_args()
    yaml_config = load_yaml_config(config_args.config)

    parser.set_defaults(**yaml_config)

    args = parser.parse_args()

    if args.data_root is None:
        parser.error("the following arguments are required: --data-root")


    set_seed(args.seed)

    data_root = Path(args.data_root)
    # Always place experiment outputs inside output_iq/
    output_root = Path("output_iq")
    requested_output = Path(args.output_dir)

    # If user already wrote output_iq/..., do not duplicate it
    if requested_output.parts and requested_output.parts[0] == output_root.name:
        output_dir = requested_output
    else:
        output_dir = output_root / requested_output

    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = args.classes
    print("Classes:", class_names)

    files, labels = collect_files(data_root, class_names)

    print(f"Total IQ files: {len(files)}")

    full_ds = IQFFTDataset(
        files,
        labels,
        chunk_size=args.chunk_size,
        nfft=args.nfft,
        max_chunks_per_file=args.max_chunks_per_file,
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

    write_experiment_config(
        args,
        output_dir,
        class_names,
        len(train_ds),
        len(valid_ds),
        total_files=len(files),
    )


    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = SmallFFT1DCNN(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_valid_acc = 0.0
    best_epoch = -1

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
            torch.save(model.state_dict(), output_dir / "best_fft_1dcnn.pt")

            save_confusion_matrix(
                y_true,
                y_pred,
                class_names,
                output_dir / "confusion_matrix.png",
            )

            with open(output_dir / "classification_report.txt", "w") as f:
                f.write(
                    classification_report(
                        y_true,
                        y_pred,
                        target_names=class_names,
                    )
                )
                
    write_experiment_config(
        args,
        output_dir,
        class_names,
        len(train_ds),
        len(valid_ds),
        total_files=len(files),
        best_valid_acc=best_valid_acc,
        best_epoch=best_epoch,
    )

    print()
    print("Done.")
    print(f"Best valid accuracy: {best_valid_acc:.4f}")
    print(f"Saved model to: {output_dir / 'best_fft_1dcnn.pt'}")
    print(f"Saved confusion matrix to: {output_dir / 'confusion_matrix.png'}")
    print(f"Saved report to: {output_dir / 'classification_report.txt'}")


if __name__ == "__main__":
    main()