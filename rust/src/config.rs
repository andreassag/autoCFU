//! Configuration schemas, YAML deserialization, and parameter validation.

use std::fs::File;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{AutoCFUError, Result};

/// Supported ONNX model families in AutoCFU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    /// Automatically infer the model type from ONNX output tensor names/metadata.
    #[default]
    Auto,
    /// Radial CenterNet colony detector.
    Centernet,
    /// 2D StarDist star-convex polygon instance segmentation model.
    Stardist,
    /// 3-class Plate & Section Countability classifier.
    Countability,
}

impl ModelType {
    /// Parse a case-insensitive model type name.
    ///
    /// # Errors
    /// Returns [`AutoCFUError::InvalidConfig`] if `raw` is not one of
    /// `auto`, `centernet`, `stardist`, or `countability`.
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "centernet" => Ok(Self::Centernet),
            "stardist" => Ok(Self::Stardist),
            "countability" => Ok(Self::Countability),
            other => Err(AutoCFUError::InvalidConfig(format!(
                "Unsupported model_type '{other}'; expected auto, centernet, stardist, or countability"
            ))),
        }
    }

    /// Return the canonical lowercase name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Centernet => "centernet",
            Self::Stardist => "stardist",
            Self::Countability => "countability",
        }
    }
}

/// Cross-tile merging algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MergerType {
    /// Greedy Non-Maximum Merging with confidence-weighted coordinate/ray averaging.
    #[default]
    GreedyNmm,
    /// Standard Non-Maximum Suppression.
    Nms,
    /// Disable cross-tile duplicate merging.
    None,
}

impl MergerType {
    /// Parse a merger strategy name (`greedy_nmm`, `nms`, `none`).
    ///
    /// # Errors
    /// Returns [`AutoCFUError::InvalidConfig`] for unrecognized merger names.
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "greedy_nmm" => Ok(Self::GreedyNmm),
            "nms" => Ok(Self::Nms),
            "none" => Ok(Self::None),
            other => Err(AutoCFUError::InvalidConfig(format!(
                "Unsupported merger_type '{other}'; expected greedy_nmm, nms, or none"
            ))),
        }
    }
}

/// Geometric overlap or proximity metric used for merging detections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MatchMetric {
    /// Euclidean distance between centers normalized by `min(r_a, r_b)`.
    #[default]
    CenterDist,
    /// Intersection over Smaller bounding-box area.
    Ios,
    /// Intersection over Union of bounding boxes.
    Iou,
}

impl MatchMetric {
    /// Parse a match metric name (`center_dist`, `ios`, `iou`).
    ///
    /// # Errors
    /// Returns [`AutoCFUError::InvalidConfig`] for unrecognized metric names.
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "center_dist" => Ok(Self::CenterDist),
            "ios" => Ok(Self::Ios),
            "iou" => Ok(Self::Iou),
            other => Err(AutoCFUError::InvalidConfig(format!(
                "Unsupported match_metric '{other}'; expected center_dist, ios, or iou"
            ))),
        }
    }
}

/// Evaluation scope when running a Countability classifier model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CountabilityScope {
    /// Evaluate both the full dish image and each sliced 640x640 section.
    #[default]
    Both,
    /// Evaluate only the resized whole-plate image.
    Plate,
    /// Evaluate only 640x640 sliding-window plate sections.
    Sections,
}

impl CountabilityScope {
    /// Parse a countability scope name (`both`, `plate`, `sections`).
    ///
    /// # Errors
    /// Returns [`AutoCFUError::InvalidConfig`] for unrecognized scope strings.
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "both" => Ok(Self::Both),
            "plate" => Ok(Self::Plate),
            "sections" => Ok(Self::Sections),
            other => Err(AutoCFUError::InvalidConfig(format!(
                "Unsupported countability_scope '{other}'; expected both, plate, or sections"
            ))),
        }
    }
}

/// ONNX Runtime execution provider preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DevicePreference {
    /// Prefer GPU if available and fall back to CPU automatically.
    #[default]
    Auto,
    /// Run strictly on CPU.
    Cpu,
    /// Request CUDA execution provider.
    Cuda,
}

impl DevicePreference {
    /// Parse a device preference name (`auto`, `cpu`, `cuda`).
    ///
    /// # Errors
    /// Returns [`AutoCFUError::InvalidConfig`] for unrecognized values.
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            other => Err(AutoCFUError::InvalidConfig(format!(
                "Unsupported device '{other}'; expected auto, cpu, or cuda"
            ))),
        }
    }
}

/// 2D sliding-window dish slicing configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SlicingConfig {
    /// Width of each slice tile in pixels.
    pub slice_width: usize,
    /// Height of each slice tile in pixels.
    pub slice_height: usize,
    /// Fractional horizontal overlap between adjacent tiles in `[0.0, 1.0)`.
    pub overlap_width_ratio: f64,
    /// Fractional vertical overlap between adjacent tiles in `[0.0, 1.0)`.
    pub overlap_height_ratio: f64,
}

impl Default for SlicingConfig {
    fn default() -> Self {
        Self {
            slice_width: 640,
            slice_height: 640,
            overlap_width_ratio: 0.20,
            overlap_height_ratio: 0.20,
        }
    }
}

impl SlicingConfig {
    /// Validate slicing dimensions and overlap ratios.
    ///
    /// # Errors
    /// Returns [`AutoCFUError::InvalidConfig`] if dimensions are zero or overlaps are outside `[0.0, 1.0)`.
    pub fn validate(&self) -> Result<()> {
        if self.slice_width == 0 || self.slice_height == 0 {
            return Err(AutoCFUError::InvalidConfig(format!(
                "Slice dimensions must be positive, got {}x{}",
                self.slice_width, self.slice_height
            )));
        }
        if !(0.0..1.0).contains(&self.overlap_width_ratio) {
            return Err(AutoCFUError::InvalidConfig(format!(
                "overlap_width_ratio must be in [0.0, 1.0), got {}",
                self.overlap_width_ratio
            )));
        }
        if !(0.0..1.0).contains(&self.overlap_height_ratio) {
            return Err(AutoCFUError::InvalidConfig(format!(
                "overlap_height_ratio must be in [0.0, 1.0), got {}",
                self.overlap_height_ratio
            )));
        }
        Ok(())
    }
}

/// Inference, filtering, merging, and countability configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceConfig {
    /// Model architecture family (`auto`, `centernet`, `stardist`, `countability`).
    pub model_type: ModelType,
    /// Square model input dimension in pixels.
    pub input_size: usize,
    /// Execution provider preference.
    pub device: DevicePreference,
    /// Minimum confidence score to keep predictions.
    pub confidence_threshold: f64,
    /// Cross-tile merging threshold.
    pub postprocess_match_threshold: f64,
    /// Cross-tile merging strategy.
    pub merger_type: MergerType,
    /// Geometric metric used for duplicate suppression/merging.
    pub match_metric: MatchMetric,
    /// Intra-tile NMS match threshold before global coordinate merging.
    pub intra_tile_nms_threshold: f64,
    /// Minimum valid radial colony size in pixels.
    pub min_radius: f64,
    /// Maximum valid radial colony size in pixels.
    pub max_radius: f64,
    /// Minimum valid bounding-box width or height in pixels.
    pub min_box_size: f64,
    /// Maximum valid bounding-box width or height in pixels.
    pub max_box_size: f64,
    /// Maximum allowed bounding-box aspect ratio.
    pub max_aspect_ratio: f64,
    /// Expected foreground class ID.
    pub target_class_id: Option<i64>,
    /// Label text assigned to detected colonies.
    pub class_name: String,
    /// Top-K candidates kept during raw-map decoding fallback.
    pub top_k: usize,
    /// Expected number of equiangular rays for 2D StarDist models.
    pub num_rays: usize,
    /// Evaluation scope for countability classifier models.
    pub countability_scope: CountabilityScope,
    /// Uncountable class probability threshold for binary countability.
    pub uncountable_threshold: f64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::Auto,
            input_size: 640,
            device: DevicePreference::Auto,
            confidence_threshold: 0.35,
            postprocess_match_threshold: 0.50,
            merger_type: MergerType::GreedyNmm,
            match_metric: MatchMetric::CenterDist,
            intra_tile_nms_threshold: 0.45,
            min_radius: 1.5,
            max_radius: 250.0,
            min_box_size: 3.0,
            max_box_size: 500.0,
            max_aspect_ratio: 4.0,
            target_class_id: Some(0),
            class_name: "colony".to_string(),
            top_k: 300,
            num_rays: 32,
            countability_scope: CountabilityScope::Both,
            uncountable_threshold: 0.50,
        }
    }
}

impl InferenceConfig {
    /// Validate inference configuration bounds.
    ///
    /// # Errors
    /// Returns [`AutoCFUError::InvalidConfig`] when any threshold or size bound is out of range.
    pub fn validate(&self) -> Result<()> {
        if self.input_size == 0 {
            return Err(AutoCFUError::InvalidConfig(
                "input_size must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.confidence_threshold) {
            return Err(AutoCFUError::InvalidConfig(format!(
                "confidence_threshold must be in [0.0, 1.0], got {}",
                self.confidence_threshold
            )));
        }
        if !(0.0..=2.0).contains(&self.postprocess_match_threshold) {
            return Err(AutoCFUError::InvalidConfig(format!(
                "postprocess_match_threshold must be in [0.0, 2.0], got {}",
                self.postprocess_match_threshold
            )));
        }
        if !(0.0..=2.0).contains(&self.intra_tile_nms_threshold) {
            return Err(AutoCFUError::InvalidConfig(format!(
                "intra_tile_nms_threshold must be in [0.0, 2.0], got {}",
                self.intra_tile_nms_threshold
            )));
        }
        if self.min_radius < 0.0 || self.max_radius <= self.min_radius {
            return Err(AutoCFUError::InvalidConfig(format!(
                "Invalid radius bounds: min_radius={}, max_radius={}",
                self.min_radius, self.max_radius
            )));
        }
        if self.min_box_size < 0.0 || self.max_box_size <= self.min_box_size {
            return Err(AutoCFUError::InvalidConfig(format!(
                "Invalid box size bounds: min_box_size={}, max_box_size={}",
                self.min_box_size, self.max_box_size
            )));
        }
        if self.max_aspect_ratio < 1.0 {
            return Err(AutoCFUError::InvalidConfig(format!(
                "max_aspect_ratio must be >= 1.0, got {}",
                self.max_aspect_ratio
            )));
        }
        if self.top_k == 0 || self.num_rays == 0 {
            return Err(AutoCFUError::InvalidConfig(
                "top_k and num_rays must both be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.uncountable_threshold) {
            return Err(AutoCFUError::InvalidConfig(format!(
                "uncountable_threshold must be in [0.0, 1.0], got {}",
                self.uncountable_threshold
            )));
        }
        Ok(())
    }
}

/// Top-level configuration bundling slicing and inference parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct EngineConfig {
    /// Slicing configuration.
    pub slicing: SlicingConfig,
    /// Inference configuration.
    pub inference: InferenceConfig,
}

impl EngineConfig {
    /// Validate both slicing and inference sections.
    ///
    /// # Errors
    /// Returns [`AutoCFUError::InvalidConfig`] if any nested setting is invalid.
    pub fn validate(&self) -> Result<()> {
        self.slicing.validate()?;
        self.inference.validate()?;
        Ok(())
    }

    /// Load configuration from an optional YAML file path (or defaults when `None`).
    ///
    /// # Errors
    /// Returns [`AutoCFUError::FileNotFound`], [`AutoCFUError::Yaml`], or [`AutoCFUError::InvalidConfig`].
    pub fn load_yaml(path: Option<&Path>) -> Result<Self> {
        let Some(cfg_path) = path else {
            return Ok(Self::default());
        };
        if !cfg_path.exists() {
            return Err(AutoCFUError::FileNotFound(cfg_path.to_path_buf()));
        }
        let file = File::open(cfg_path)?;
        let cfg: Self = serde_yaml::from_reader(file)?;
        cfg.validate()?;
        Ok(cfg)
    }
}
