"""CPU-compatible inference restricted to images in a prepared research manifest."""

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.model import load_checkpoint
from ml.train import ManifestDataset, collect_logits


def infer_manifest(
    source: Path,
    manifest: Path,
    checkpoint_path: Path,
    split: str,
    device_name: str = "cpu",
) -> list[dict[str, object]]:
    """Run calibrated research inference only on the prepared labeled manifest."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test.")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda, or auto.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(device_name)
    model, checkpoint = load_checkpoint(str(checkpoint_path), device)
    class_names = tuple(checkpoint["class_order"])
    dataset = ManifestDataset(source, manifest, split, class_names)
    with manifest.open(newline="", encoding="utf-8") as file:
        manifest_rows = {
            row["path"]: row for row in csv.DictReader(file) if row["split"] == split
        }
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    logits, labels, paths = collect_logits(model, loader, device)
    probabilities = torch.softmax(logits / float(checkpoint["temperature"]), dim=1)
    predictions = probabilities.argmax(dim=1)

    results = []
    for index, path in enumerate(paths):
        if path not in manifest_rows:
            raise RuntimeError(f"Dataset row is absent from the prepared manifest: {path}")
        source_path = (source / path).resolve()
        try:
            source_path.relative_to(source.resolve())
        except ValueError as exc:
            raise ValueError(f"Manifest image path escapes the source directory: {path}") from exc
        results.append(
            {
                "path": path,
                "actual": class_names[int(labels[index])],
                "predicted": class_names[int(predictions[index])],
                "model_version": checkpoint["model_version"],
                "probabilities": {
                    class_name: float(probabilities[index, class_index])
                    for class_index, class_name in enumerate(class_names)
                },
                "probabilities_calibrated_on_validation": True,
            }
        )
    return results


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--acknowledge-research-only",
        action="store_true",
        help="Required: scores are not diagnoses and do not verify image modality.",
    )
    args = parser.parse_args()
    if not args.acknowledge_research_only:
        parser.error(
            "This offline command is for curated research manifests only. "
            "Pass --acknowledge-research-only after reviewing the limitations."
        )
    results = infer_manifest(
        args.source, args.manifest, args.checkpoint, args.split, args.device
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
