"""Command-line interface for the AutoCFU colony detection engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .pipeline import AutoCFUEngine, InferenceResult, export_batch_summary_csv

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autocfu",
        description="AutoCFU: Run high-resolution sliced inference for microbial colony detection and counting.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=str,
        help="Path to a single agar plate image file.",
    )
    input_group.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing plate images for batch processing.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the exported RT-DETRv4 ONNX model file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to inference YAML configuration file (defaults to configs/default_inference.yaml if present).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Override detection confidence threshold (e.g. 0.35).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="Override slice overlap ratio (e.g. 0.20 for 20%% overlap).",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=None,
        help="Override slice window width and height (e.g. 640).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory to save detection results and overlays (default: results).",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="Save annotated images with colony bounding boxes.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save structured JSON detection coordinates for each image.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save CSV summary and detection tables.",
    )
    parser.add_argument(
        "--merger",
        type=str,
        default=None,
        choices=["greedy_nmm", "nms"],
        help="Override bounding box merger type (greedy_nmm or nms).",
    )
    parser.add_argument(
        "--match-metric",
        type=str,
        default=None,
        choices=["iou", "ios"],
        help="Override box matching metric (iou or ios).",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=None,
        help="Override postprocess match IoU/IoS threshold (e.g. 0.50).",
    )
    parser.add_argument(
        "--intra-tile-threshold",
        type=float,
        default=None,
        help="Override intra-tile NMS duplicate query suppression threshold (e.g. 0.45, 0 to disable).",
    )
    parser.add_argument(
        "--intra-tile-metric",
        type=str,
        default=None,
        choices=["iou", "ios"],
        help="Override intra-tile match metric (iou or ios).",
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=None,
        help="Override minimum box width and height in pixels (e.g. 6.0, 0 to disable).",
    )
    parser.add_argument(
        "--max-box-size",
        type=float,
        default=None,
        help="Override maximum box width and height in pixels (e.g. 300.0, 0 to disable).",
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=None,
        help="Override maximum allowed aspect ratio max(w/h, h/w) (e.g. 2.5, 0 to disable).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution provider for ONNX Runtime (default: auto).",
    )

    return parser


def find_default_config() -> str | None:
    """Search common locations for configs/default_inference.yaml."""
    candidates = [
        Path("configs/default_inference.yaml"),
        Path("../configs/default_inference.yaml"),
        Path(__file__).resolve().parent.parent.parent / "configs" / "default_inference.yaml",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def main(args_list: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(args_list)

    # 1. Resolve configuration
    config_path = args.config or find_default_config()
    config = load_config(config_path)

    # Apply CLI overrides
    if args.confidence is not None:
        config.inference.confidence_threshold = args.confidence
    if args.overlap is not None:
        config.slicing.overlap_width_ratio = args.overlap
        config.slicing.overlap_height_ratio = args.overlap
    if args.slice_size is not None:
        config.slicing.slice_width = args.slice_size
        config.slicing.slice_height = args.slice_size
    if args.merger is not None:
        config.inference.merger_type = args.merger
    if args.match_metric is not None:
        config.inference.match_metric = args.match_metric
    if args.match_threshold is not None:
        config.inference.postprocess_match_threshold = args.match_threshold
    if args.intra_tile_threshold is not None:
        config.inference.intra_tile_nms_threshold = args.intra_tile_threshold
    if args.intra_tile_metric is not None:
        config.inference.intra_tile_match_metric = args.intra_tile_metric
    if args.min_box_size is not None:
        config.inference.min_box_size = args.min_box_size
    if args.max_box_size is not None:
        config.inference.max_box_size = args.max_box_size
    if args.max_aspect_ratio is not None:
        config.inference.max_aspect_ratio = args.max_aspect_ratio

    # 2. Gather image paths
    image_paths: list[Path] = []
    if args.image:
        p = Path(args.image)
        if not p.is_file():
            print(f"Error: Image file not found: {p}", file=sys.stderr)
            return 1
        image_paths.append(p)
    elif args.input_dir:
        dir_p = Path(args.input_dir)
        if not dir_p.is_dir():
            print(f"Error: Input directory not found: {dir_p}", file=sys.stderr)
            return 1
        for f in sorted(dir_p.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(f)

    if not image_paths:
        print("No valid images found to process.", file=sys.stderr)
        return 1

    # 3. Initialize engine
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        engine = AutoCFUEngine(
            model_path=args.model,
            config=config,
            device=args.device,
        )
    except Exception as err:
        print(f"Failed to initialize inference engine: {err}", file=sys.stderr)
        return 1

    print(f"Starting inference on {len(image_paths)} plate image(s)...")
    print(
        f"Settings: slice={config.slicing.slice_width}x{config.slicing.slice_height}, "
        f"overlap={config.slicing.overlap_width_ratio * 100:.0f}%, "
        f"conf_thresh={config.inference.confidence_threshold:.2f}, "
        f"merger={config.inference.merger_type}"
    )

    results: list[InferenceResult] = []
    total_colonies = 0

    for idx, img_path in enumerate(image_paths, start=1):
        try:
            res = engine.infer_image(img_path)
            results.append(res)
            total_colonies += res.colony_count

            print(
                f"[{idx}/{len(image_paths)}] {img_path.name}: "
                f"{res.colony_count} colonies ({res.num_slices} tiles, {res.execution_time_seconds:.2f}s)"
            )

            stem = img_path.stem
            if args.save_overlay:
                overlay_path = output_dir / f"{stem}_detected.jpg"
                res.draw_overlay(output_path=overlay_path)

            if args.save_json:
                json_path = output_dir / f"{stem}_detections.json"
                res.save_json(json_path)

            if args.save_csv:
                csv_path = output_dir / f"{stem}_detections.csv"
                res.save_csv(csv_path)

        except Exception as err:
            print(f"Error processing {img_path}: {err}", file=sys.stderr)

    if args.save_csv and len(results) > 1:
        batch_summary_csv = output_dir / "summary.csv"
        export_batch_summary_csv(results, batch_summary_csv)
        print(f"Summary table saved to {batch_summary_csv}")

    print(f"\nDone. Processed {len(results)} image(s), detected {total_colonies} colonies total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
