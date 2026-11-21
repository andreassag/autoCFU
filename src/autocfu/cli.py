"""Command-line interface for AutoCFU inference across CenterNet, StarDist, and Countability ONNX models."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from autocfu.config import EngineConfig, load_config
from autocfu.pipeline import IMAGE_EXTENSIONS, AutoCFUEngine, export_batch_summary_csv


def find_default_config() -> Path | None:
    """Locate ``configs/default_inference.yaml`` relative to working directory or package root."""
    candidates = [
        Path.cwd() / "configs" / "default_inference.yaml",
        Path(__file__).resolve().parent.parent.parent / "configs" / "default_inference.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``autocfu`` CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="autocfu",
        description=(
            "AutoCFU standalone ONNX inference engine for Petri dish colony detection "
            "(Radial CenterNet), instance segmentation (2D StarDist), and countability classification."
        ),
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=Path, help="Path to a single Petri dish image.")
    input_group.add_argument("--input-dir", type=Path, help="Directory containing multiple Petri dish images.")

    parser.add_argument("--model", type=Path, required=True, help="Path to exported .onnx model file.")
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["auto", "centernet", "stardist", "countability"],
        help="Model architecture type (default: auto-detected from ONNX outputs).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file (defaults to configs/default_inference.yaml if found).",
    )
    parser.add_argument("--confidence", type=float, default=None, help="Confidence score threshold in [0.0, 1.0].")
    parser.add_argument("--slice-size", type=int, default=None, help="Square slice dimension in pixels (e.g., 640).")
    parser.add_argument("--overlap", type=float, default=None, help="Slice overlap ratio in [0.0, 1.0).")
    parser.add_argument(
        "--merger",
        type=str,
        default=None,
        choices=["greedy_nmm", "nms", "none"],
        help="Cross-tile duplicate merging algorithm.",
    )
    parser.add_argument(
        "--match-metric",
        type=str,
        default=None,
        choices=["center_dist", "ios", "iou"],
        help="Matching metric used during cross-tile merging.",
    )
    parser.add_argument("--match-threshold", type=float, default=None, help="Threshold for merging overlapping boxes.")
    parser.add_argument("--min-radius", type=float, default=None, help="Minimum colony radius in pixels.")
    parser.add_argument("--max-radius", type=float, default=None, help="Maximum colony radius in pixels.")
    parser.add_argument(
        "--countability-scope",
        type=str,
        default=None,
        choices=["both", "plate", "sections"],
        help="Countability evaluation mode: whole plate, sliced 640x640 sections, or both.",
    )
    parser.add_argument(
        "--uncountable-threshold",
        type=float,
        default=None,
        help="Probability threshold on class 'uncountable' for binary countability verdict.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="ONNX Runtime execution provider preference.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for output artifacts.")
    parser.add_argument("--save-overlay", action="store_true", help="Save rendered detection/countability overlays.")
    parser.add_argument("--save-json", action="store_true", help="Save per-image JSON predictions.")
    parser.add_argument("--save-csv", action="store_true", help="Save per-image CSV predictions (plus batch summary).")
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Save 16-bit uint16 instance segmentation masks (for CenterNet/StarDist).",
    )
    parser.add_argument("--show-labels", action="store_true", help="Draw confidence scores on saved overlay images.")
    return parser


def _apply_cli_overrides(config: EngineConfig, args: argparse.Namespace) -> EngineConfig:
    if args.model_type is not None:
        config.inference.model_type = args.model_type
    if args.confidence is not None:
        config.inference.confidence_threshold = args.confidence
    if args.slice_size is not None:
        config.slicing.slice_width = args.slice_size
        config.slicing.slice_height = args.slice_size
        config.inference.input_size = args.slice_size
    if args.overlap is not None:
        config.slicing.overlap_width_ratio = args.overlap
        config.slicing.overlap_height_ratio = args.overlap
    if args.merger is not None:
        config.inference.merger_type = args.merger
    if args.match_metric is not None:
        config.inference.match_metric = args.match_metric
    if args.match_threshold is not None:
        config.inference.postprocess_match_threshold = args.match_threshold
    if args.min_radius is not None:
        config.inference.min_radius = args.min_radius
    if args.max_radius is not None:
        config.inference.max_radius = args.max_radius
    if args.countability_scope is not None:
        config.inference.countability_scope = args.countability_scope
    if args.uncountable_threshold is not None:
        config.inference.uncountable_threshold = args.uncountable_threshold
    if args.device is not None:
        config.inference.device = args.device

    # Re-validate mutated fields
    config.slicing.__post_init__()
    config.inference.__post_init__()
    return config


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point returning an OS exit status code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg_path = args.config if args.config is not None else find_default_config()
    config = _apply_cli_overrides(load_config(cfg_path), args)

    if args.image is not None:
        if not args.image.exists():
            print(f"Error: Input image not found: {args.image}", file=sys.stderr)
            return 1
        image_paths = [args.image]
    else:
        assert args.input_dir is not None
        if not args.input_dir.is_dir():
            print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
            return 1
        image_paths = sorted(
            p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            print(f"Error: No image files found in directory: {args.input_dir}", file=sys.stderr)
            return 1

    engine = AutoCFUEngine(model_path=args.model, config=config)
    results = engine.predict_batch(
        image_paths,
        output_dir=args.output_dir,
        save_overlay=args.save_overlay,
        save_json=args.save_json,
        save_csv=args.save_csv,
        save_mask=args.save_mask,
        show_labels=args.show_labels,
    )

    if args.save_csv and len(results) > 1:
        summary_path = Path(args.output_dir) / "batch_summary.csv"
        export_batch_summary_csv(results, summary_path)

    for res in results:
        name = Path(res.image_path).name if res.image_path else "<memory>"
        if res.model_type == "countability" and res.countability_report is not None:
            rep = res.countability_report
            verdict = "COUNTABLE" if rep.is_countable else "UNCOUNTABLE"
            wp_label = rep.whole_plate.pred_label if rep.whole_plate is not None else "n/a"
            print(
                f"{name}: verdict={verdict} (whole_plate={wp_label}, "
                f"sections=empty:{rep.empty_section_count}/countable:{rep.countable_section_count}/"
                f"uncountable:{rep.uncountable_section_count}) [{res.inference_time_ms:.1f} ms]"
            )
        else:
            print(
                f"{name}: {res.colony_count} colonies "
                f"(model={res.model_type}, slices={res.num_slices}, {res.inference_time_ms:.1f} ms)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
