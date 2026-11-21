//! Command-line binary for AutoCFU Rust ONNX inference.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;

use autocfu::{
    export_batch_summary_csv, AutoCFUEngine, CountabilityScope, DevicePreference, EngineConfig,
    InferenceResult, MatchMetric, MergerType, ModelType, IMAGE_EXTENSIONS,
};

#[derive(Parser, Debug)]
#[command(
    name = "autocfu",
    about = "AutoCFU standalone Rust ONNX inference engine (CenterNet, 2D StarDist, Countability)"
)]
struct CliArgs {
    /// Path to a single Petri dish image.
    #[arg(
        long,
        conflicts_with = "input_dir",
        required_unless_present = "input_dir"
    )]
    image: Option<PathBuf>,

    /// Directory containing Petri dish images.
    #[arg(long, conflicts_with = "image", required_unless_present = "image")]
    input_dir: Option<PathBuf>,

    /// Path to exported .onnx model weights.
    #[arg(long)]
    model: PathBuf,

    /// Model family override (`auto`, `centernet`, `stardist`, `countability`).
    #[arg(long)]
    model_type: Option<String>,

    /// Optional YAML configuration file path.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Minimum confidence score threshold.
    #[arg(long)]
    confidence: Option<f64>,

    /// Square slice window dimension in pixels.
    #[arg(long)]
    slice_size: Option<usize>,

    /// Overlap ratio between neighboring tiles in [0.0, 1.0).
    #[arg(long)]
    overlap: Option<f64>,

    /// Duplicate merging strategy (`greedy_nmm`, `nms`, `none`).
    #[arg(long)]
    merger: Option<String>,

    /// Similarity metric (`center_dist`, `ios`, `iou`).
    #[arg(long)]
    match_metric: Option<String>,

    /// Threshold applied by the cross-tile merger.
    #[arg(long)]
    match_threshold: Option<f64>,

    /// Minimum radial colony size in pixels.
    #[arg(long)]
    min_radius: Option<f64>,

    /// Maximum radial colony size in pixels.
    #[arg(long)]
    max_radius: Option<f64>,

    /// Countability scope (`both`, `plate`, `sections`).
    #[arg(long)]
    countability_scope: Option<String>,

    /// Probability threshold on class `uncountable`.
    #[arg(long)]
    uncountable_threshold: Option<f64>,

    /// Execution device preference (`auto`, `cpu`, `cuda`).
    #[arg(long)]
    device: Option<String>,

    /// Output artifact directory.
    #[arg(long, default_value = "outputs")]
    output_dir: PathBuf,

    /// Save rendered visualization overlay images.
    #[arg(long)]
    save_overlay: bool,

    /// Save per-image JSON prediction artifacts.
    #[arg(long)]
    save_json: bool,

    /// Save per-image CSV predictions (and batch_summary.csv for directories).
    #[arg(long)]
    save_csv: bool,

    /// Save 16-bit uint16 TIFF instance masks for spatial colony models.
    #[arg(long)]
    save_mask: bool,
}

fn find_default_config() -> Option<PathBuf> {
    let cwd_cfg = PathBuf::from("configs/default_inference.yaml");
    if cwd_cfg.is_file() {
        return Some(cwd_cfg);
    }
    None
}

fn run_cli() -> autocfu::Result<()> {
    let args = CliArgs::parse();
    let cfg_path = args.config.or_else(find_default_config);
    let mut cfg = EngineConfig::load_yaml(cfg_path.as_deref())?;

    if let Some(ref mt) = args.model_type {
        cfg.inference.model_type = ModelType::parse(mt)?;
    }
    if let Some(conf) = args.confidence {
        cfg.inference.confidence_threshold = conf;
    }
    if let Some(ss) = args.slice_size {
        cfg.slicing.slice_width = ss;
        cfg.slicing.slice_height = ss;
        cfg.inference.input_size = ss;
    }
    if let Some(ov) = args.overlap {
        cfg.slicing.overlap_width_ratio = ov;
        cfg.slicing.overlap_height_ratio = ov;
    }
    if let Some(ref mg) = args.merger {
        cfg.inference.merger_type = MergerType::parse(mg)?;
    }
    if let Some(ref mm) = args.match_metric {
        cfg.inference.match_metric = MatchMetric::parse(mm)?;
    }
    if let Some(mt) = args.match_threshold {
        cfg.inference.postprocess_match_threshold = mt;
    }
    if let Some(mr) = args.min_radius {
        cfg.inference.min_radius = mr;
    }
    if let Some(xr) = args.max_radius {
        cfg.inference.max_radius = xr;
    }
    if let Some(ref cs) = args.countability_scope {
        cfg.inference.countability_scope = CountabilityScope::parse(cs)?;
    }
    if let Some(ut) = args.uncountable_threshold {
        cfg.inference.uncountable_threshold = ut;
    }
    if let Some(ref dev) = args.device {
        cfg.inference.device = DevicePreference::parse(dev)?;
    }

    let image_paths: Vec<PathBuf> = if let Some(ref single) = args.image {
        vec![single.clone()]
    } else if let Some(ref dir) = args.input_dir {
        if !dir.is_dir() {
            return Err(autocfu::AutoCFUError::FileNotFound(dir.clone()));
        }
        let mut found = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                let ext = path
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(str::to_ascii_lowercase)
                    .unwrap_or_default();
                if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
                    found.push(path);
                }
            }
        }
        found.sort();
        if found.is_empty() {
            return Err(autocfu::AutoCFUError::InvalidConfig(format!(
                "No supported image files found in {}",
                dir.display()
            )));
        }
        found
    } else {
        Vec::new()
    };

    let mut engine = AutoCFUEngine::new(&args.model, cfg)?;
    let mut results: Vec<InferenceResult> = Vec::with_capacity(image_paths.len());

    let any_export = args.save_overlay || args.save_json || args.save_csv || args.save_mask;
    if any_export {
        std::fs::create_dir_all(&args.output_dir)?;
    }

    for img_path in &image_paths {
        let res = engine.predict_file(img_path)?;
        let stem = img_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("dish");

        if args.save_overlay {
            let dest = args.output_dir.join(format!("{stem}_overlay.png"));
            res.draw_overlay(None, Some(&dest))?;
        }
        if args.save_json {
            let dest = args.output_dir.join(format!("{stem}_predictions.json"));
            res.save_json(dest)?;
        }
        if args.save_csv {
            let dest = args.output_dir.join(format!("{stem}_predictions.csv"));
            res.save_csv(dest)?;
        }
        if args.save_mask && res.model_type != ModelType::Countability {
            let dest = args.output_dir.join(format!("{stem}_mask.tif"));
            res.save_mask(dest)?;
        }

        let name = Path::new(img_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("dish");
        if let Some(ref rep) = res.countability {
            let verdict = if rep.is_countable {
                "COUNTABLE"
            } else {
                "UNCOUNTABLE"
            };
            let wp_lbl = rep
                .whole_plate
                .as_ref()
                .map_or("n/a", |wp| wp.pred_label.as_str());
            println!(
                "{name}: verdict={verdict} (whole_plate={wp_lbl}, sections=empty:{}/countable:{}/uncountable:{}) [{:.1} ms]",
                rep.section_summary.empty,
                rep.section_summary.countable,
                rep.section_summary.uncountable,
                res.inference_time_ms
            );
        } else {
            println!(
                "{name}: {} colonies (model={}, slices={}, {:.1} ms)",
                res.colony_count,
                res.model_type.as_str(),
                res.num_slices,
                res.inference_time_ms
            );
        }
        results.push(res);
    }

    if args.save_csv && results.len() > 1 {
        export_batch_summary_csv(&results, args.output_dir.join("batch_summary.csv"))?;
    }

    Ok(())
}

fn main() -> ExitCode {
    if let Err(err) = run_cli() {
        eprintln!("Error: {err}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
