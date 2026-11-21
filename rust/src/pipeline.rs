//! Full-dish sliced inference pipeline, uint16 instance mask export, overlays, and reports.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;

use image::{DynamicImage, ImageBuffer, Luma, Rgb, RgbImage};
use serde::Serialize;
use tiff::encoder::colortype::Gray16;
use tiff::encoder::TiffEncoder;

use crate::config::{CountabilityScope, EngineConfig, ModelType};
use crate::error::{AutoCFUError, Result};
use crate::merger::{apply_merger, Detection};
use crate::model::{AutoDetectedModel, CountabilityResult};
use crate::slicer::PlateSlicer;

/// Valid input image extensions for directory batch inference.
pub const IMAGE_EXTENSIONS: [&str; 7] = ["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"];

/// Summary counts for section-level countability predictions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SectionSummary {
    /// Total evaluated sections.
    pub total_sections: usize,
    /// Number of sections classified as `empty`.
    pub empty: usize,
    /// Number of sections classified as `countable`.
    pub countable: usize,
    /// Number of sections classified as `uncountable`.
    pub uncountable: usize,
}

/// Aggregated whole-plate and section-level countability assessment.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PlateCountabilityReport {
    /// Whole-plate binary countability decision.
    pub is_countable: bool,
    /// Probability threshold applied to `P(uncountable)`.
    pub uncountable_threshold: f64,
    /// Classification result for the resized full dish image, if evaluated.
    pub whole_plate: Option<CountabilityResult>,
    /// Aggregated counts across sections.
    pub section_summary: SectionSummary,
    /// Per-section classification predictions.
    pub sections: Vec<CountabilityResult>,
}

impl PlateCountabilityReport {
    /// Construct a report from optional whole-plate and per-section predictions.
    #[must_use]
    pub fn from_parts(
        whole_plate: Option<CountabilityResult>,
        sections: Vec<CountabilityResult>,
        uncountable_threshold: f64,
    ) -> Self {
        let mut empty = 0usize;
        let mut countable = 0usize;
        let mut uncountable = 0usize;
        for sec in &sections {
            match sec.pred_label.as_str() {
                "empty" => empty += 1,
                "countable" => countable += 1,
                "uncountable" => uncountable += 1,
                _ => {}
            }
        }

        let is_countable = if let Some(ref wp) = whole_plate {
            wp.is_countable
        } else if sections.is_empty() {
            true
        } else {
            let mean_unc = sections
                .iter()
                .map(|s| s.uncountable_probability)
                .sum::<f64>()
                / (sections.len() as f64);
            mean_unc < uncountable_threshold
        };

        Self {
            is_countable,
            uncountable_threshold,
            whole_plate,
            section_summary: SectionSummary {
                total_sections: sections.len(),
                empty,
                countable,
                uncountable,
            },
            sections,
        }
    }
}

/// Complete output returned by [`AutoCFUEngine::predict_image`] or [`AutoCFUEngine::predict_file`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InferenceResult {
    /// Optional source file path.
    pub image_path: Option<String>,
    /// Original dish image dimensions `[width, height]` in pixels.
    pub image_size: [usize; 2],
    /// Executed model family (`centernet`, `stardist`, or `countability`).
    pub model_type: ModelType,
    /// Total number of colonies detected or segmented.
    pub colony_count: usize,
    /// Number of tiles evaluated.
    pub num_slices: usize,
    /// End-to-end execution latency in milliseconds.
    pub inference_time_ms: f64,
    /// Countability assessment (populated for `Countability` models).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub countability: Option<PlateCountabilityReport>,
    /// Deduplicated global detections (for `Centernet` and `Stardist` models).
    pub detections: Vec<Detection>,
}

fn fill_polygon_u16(
    mask: &mut [u16],
    width: usize,
    height: usize,
    polygon: &[[f64; 2]],
    inst_id: u16,
) {
    if polygon.len() < 3 {
        return;
    }
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for pt in polygon {
        min_y = min_y.min(pt[1]);
        max_y = max_y.max(pt[1]);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let y_start = min_y.floor().max(0.0) as usize;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let y_end = max_y.ceil().min((height.saturating_sub(1)) as f64).max(0.0) as usize;

    let n = polygon.len();
    let mut nodes: Vec<f64> = Vec::with_capacity(n);
    for y in y_start..=y_end {
        if y >= height {
            break;
        }
        let py = (y as f64) + 0.5;
        nodes.clear();
        let mut j = n - 1;
        for i in 0..n {
            let yi = polygon[i][1];
            let yj = polygon[j][1];
            if (yi < py && yj >= py) || (yj < py && yi >= py) {
                let xi = polygon[i][0];
                let xj = polygon[j][0];
                let x_hit = xi + (py - yi) / (yj - yi) * (xj - xi);
                nodes.push(x_hit);
            }
            j = i;
        }
        nodes.sort_by(f64::total_cmp);
        for pair in nodes.as_chunks::<2>().0 {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let x0 = pair[0].floor().max(0.0) as usize;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let x1 = pair[1]
                .ceil()
                .min((width.saturating_sub(1)) as f64)
                .max(0.0) as usize;
            for x in x0..=x1 {
                if x < width {
                    mask[y * width + x] = inst_id;
                }
            }
        }
    }
}

fn fill_circle_u16(
    mask: &mut [u16],
    width: usize,
    height: usize,
    cx: f64,
    cy: f64,
    radius: f64,
    inst_id: u16,
) {
    if radius <= 0.0 {
        return;
    }
    let r_sq = radius * radius;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let y0 = (cy - radius).floor().max(0.0) as usize;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let y1 = (cy + radius)
        .ceil()
        .min((height.saturating_sub(1)) as f64)
        .max(0.0) as usize;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let x0 = (cx - radius).floor().max(0.0) as usize;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let x1 = (cx + radius)
        .ceil()
        .min((width.saturating_sub(1)) as f64)
        .max(0.0) as usize;

    for y in y0..=y1 {
        if y >= height {
            break;
        }
        let dy = (y as f64) + 0.5 - cy;
        for x in x0..=x1 {
            if x >= width {
                break;
            }
            let dx = (x as f64) + 0.5 - cx;
            if dx * dx + dy * dy <= r_sq {
                mask[y * width + x] = inst_id;
            }
        }
    }
}

fn draw_line_rgb(img: &mut RgbImage, x0: f64, y0: f64, x1: f64, y1: f64, color: Rgb<u8>) {
    let w = img.width() as i64;
    let h = img.height() as i64;
    #[allow(clippy::cast_possible_truncation)]
    let (mut ix0, mut iy0, ix1, iy1) = (
        x0.round() as i64,
        y0.round() as i64,
        x1.round() as i64,
        y1.round() as i64,
    );
    let dx = (ix1 - ix0).abs();
    let sx = if ix0 < ix1 { 1 } else { -1 };
    let dy = -(iy1 - iy0).abs();
    let sy = if iy0 < iy1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if ix0 >= 0 && ix0 < w && iy0 >= 0 && iy0 < h {
            #[allow(clippy::cast_sign_loss)]
            img.put_pixel(ix0 as u32, iy0 as u32, color);
        }
        if ix0 == ix1 && iy0 == iy1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            ix0 += sx;
        }
        if e2 <= dx {
            err += dx;
            iy0 += sy;
        }
    }
}

fn draw_rect_outline_rgb(
    img: &mut RgbImage,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    color: Rgb<u8>,
) {
    let fx0 = x0 as f64;
    let fy0 = y0 as f64;
    let fx1 = x1 as f64;
    let fy1 = y1 as f64;
    draw_line_rgb(img, fx0, fy0, fx1, fy0, color);
    draw_line_rgb(img, fx1, fy0, fx1, fy1, color);
    draw_line_rgb(img, fx1, fy1, fx0, fy1, color);
    draw_line_rgb(img, fx0, fy1, fx0, fy0, color);
}

fn tint_rect_rgb(
    img: &mut RgbImage,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    color: Rgb<u8>,
    alpha: f64,
) {
    let xmax = x1.min(img.width() as usize);
    let ymax = y1.min(img.height() as usize);
    let inv = 1.0 - alpha;
    for y in y0..ymax {
        for x in x0..xmax {
            #[allow(clippy::cast_possible_truncation)]
            let px = img.get_pixel_mut(x as u32, y as u32);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                px[0] = (f64::from(px[0]) * inv + f64::from(color[0]) * alpha).round() as u8;
                px[1] = (f64::from(px[1]) * inv + f64::from(color[1]) * alpha).round() as u8;
                px[2] = (f64::from(px[2]) * inv + f64::from(color[2]) * alpha).round() as u8;
            }
        }
    }
}

impl InferenceResult {
    /// Mean colony radius across all detections (0.0 when empty).
    #[must_use]
    pub fn mean_radius_px(&self) -> f64 {
        if self.detections.is_empty() {
            0.0
        } else {
            self.detections.iter().map(|d| d.radius_px).sum::<f64>()
                / (self.detections.len() as f64)
        }
    }

    /// Rasterize StarDist polygons or CenterNet circles into a 16-bit row-major instance mask (`height x width`).
    #[must_use]
    pub fn to_mask_u16(&self) -> Vec<u16> {
        let width = self.image_size[0];
        let height = self.image_size[1];
        let mut mask = vec![0_u16; width * height];

        for (idx, det) in self.detections.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let inst_id = (idx + 1).min(usize::from(u16::MAX)) as u16;
            if let Some(ref poly) = det.polygon {
                fill_polygon_u16(&mut mask, width, height, poly, inst_id);
            } else if det.radius_px > 0.0 {
                fill_circle_u16(
                    &mut mask,
                    width,
                    height,
                    det.center_xy[0],
                    det.center_xy[1],
                    det.radius_px,
                    inst_id,
                );
            }
        }
        mask
    }

    /// Save the rasterized 16-bit `uint16` instance mask as a `.tif` or `.png` image.
    ///
    /// # Errors
    /// Returns an error if creating parent directories or encoding the TIFF/PNG file fails.
    pub fn save_mask(&self, output_path: impl AsRef<Path>) -> Result<PathBuf> {
        let dest = output_path.as_ref().to_path_buf();
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mask = self.to_mask_u16();
        #[allow(clippy::cast_possible_truncation)]
        let width = self.image_size[0] as u32;
        #[allow(clippy::cast_possible_truncation)]
        let height = self.image_size[1] as u32;

        let ext = dest
            .extension()
            .and_then(|s| s.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();

        if ext == "tif" || ext == "tiff" {
            let file = File::create(&dest)?;
            let mut encoder = TiffEncoder::new(BufWriter::new(file))?;
            encoder.write_image::<Gray16>(width, height, &mask)?;
        } else {
            let img_buf: ImageBuffer<Luma<u16>, Vec<u16>> =
                ImageBuffer::from_raw(width, height, mask).ok_or_else(|| {
                    AutoCFUError::InvalidGeometry("Failed to wrap uint16 mask buffer".to_string())
                })?;
            img_buf.save(&dest)?;
        }
        Ok(dest)
    }

    /// Render colony contours, circles, or countability overlays onto an RGB dish image.
    ///
    /// # Errors
    /// Returns an error if saving the overlay file fails.
    pub fn draw_overlay(
        &self,
        base_image: Option<&DynamicImage>,
        output_path: Option<&Path>,
    ) -> Result<RgbImage> {
        let mut canvas = match base_image {
            Some(img) => img.to_rgb8(),
            None => {
                if let Some(ref src) = self.image_path {
                    if Path::new(src).exists() {
                        image::open(src)?.to_rgb8()
                    } else {
                        #[allow(clippy::cast_possible_truncation)]
                        RgbImage::new(self.image_size[0] as u32, self.image_size[1] as u32)
                    }
                } else {
                    #[allow(clippy::cast_possible_truncation)]
                    RgbImage::new(self.image_size[0] as u32, self.image_size[1] as u32)
                }
            }
        };

        if self.model_type == ModelType::Countability {
            if let Some(ref rep) = self.countability {
                for sec in &rep.sections {
                    if let Some([x0, y0, x1, y1]) = sec.region_bbox {
                        let col = match sec.pred_label.as_str() {
                            "empty" => Rgb([52, 152, 219]),
                            "countable" => Rgb([46, 204, 113]),
                            _ => Rgb([231, 76, 60]),
                        };
                        tint_rect_rgb(&mut canvas, x0, y0, x1, y1, col, 0.22);
                        draw_rect_outline_rgb(&mut canvas, x0, y0, x1, y1, col);
                    }
                }
                let banner = if rep.is_countable {
                    Rgb([39, 174, 96])
                } else {
                    Rgb([192, 57, 43])
                };
                let canvas_width = canvas.width() as usize;
                tint_rect_rgb(&mut canvas, 0, 0, canvas_width, 16, banner, 0.75);
            }
        } else {
            let outline = Rgb([0, 255, 160]);
            let center_dot = Rgb([255, 82, 82]);
            for det in &self.detections {
                if let Some(ref poly) = det.polygon {
                    if poly.len() >= 3 {
                        for idx in 0..poly.len() {
                            let next = (idx + 1) % poly.len();
                            draw_line_rgb(
                                &mut canvas,
                                poly[idx][0],
                                poly[idx][1],
                                poly[next][0],
                                poly[next][1],
                                outline,
                            );
                        }
                    }
                } else {
                    let steps = 24usize;
                    let r = det.radius_px.max(1.0);
                    let two_pi_over_steps = (2.0 * std::f64::consts::PI) / (steps as f64);
                    for idx in 0..steps {
                        let t0 = (idx as f64) * two_pi_over_steps;
                        let t1 = (((idx + 1) % steps) as f64) * two_pi_over_steps;
                        draw_line_rgb(
                            &mut canvas,
                            det.center_xy[0] + r * t0.cos(),
                            det.center_xy[1] + r * t0.sin(),
                            det.center_xy[0] + r * t1.cos(),
                            det.center_xy[1] + r * t1.sin(),
                            outline,
                        );
                    }
                }

                #[allow(clippy::cast_possible_truncation)]
                let cx = det.center_xy[0].round() as i64;
                #[allow(clippy::cast_possible_truncation)]
                let cy = det.center_xy[1].round() as i64;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let px = cx + dx;
                        let py = cy + dy;
                        if px >= 0
                            && py >= 0
                            && px < i64::from(canvas.width())
                            && py < i64::from(canvas.height())
                        {
                            #[allow(clippy::cast_sign_loss)]
                            canvas.put_pixel(px as u32, py as u32, center_dot);
                        }
                    }
                }
            }
        }

        if let Some(dest) = output_path {
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }
            canvas.save(dest)?;
        }

        Ok(canvas)
    }

    /// Serialize results to a formatted JSON file.
    ///
    /// # Errors
    /// Returns [`AutoCFUError::Io`] or [`AutoCFUError::Json`] if writing fails.
    pub fn save_json(&self, output_path: impl AsRef<Path>) -> Result<PathBuf> {
        let dest = output_path.as_ref().to_path_buf();
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(&dest)?;
        serde_json::to_writer_pretty(BufWriter::new(file), self)?;
        Ok(dest)
    }

    /// Serialize detections or countability predictions to CSV.
    ///
    /// # Errors
    /// Returns [`AutoCFUError::Io`] or [`AutoCFUError::Csv`] if writing fails.
    pub fn save_csv(&self, output_path: impl AsRef<Path>) -> Result<PathBuf> {
        let dest = output_path.as_ref().to_path_buf();
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut wtr = csv::Writer::from_path(&dest)?;

        if self.model_type == ModelType::Countability {
            if let Some(ref rep) = self.countability {
                wtr.write_record([
                    "region_type",
                    "xmin",
                    "ymin",
                    "xmax",
                    "ymax",
                    "pred_class_id",
                    "pred_label",
                    "is_countable",
                    "prob_empty",
                    "prob_countable",
                    "prob_uncountable",
                ])?;
                if let Some(ref wp) = rep.whole_plate {
                    wtr.write_record([
                        "whole_plate".to_string(),
                        "0".to_string(),
                        "0".to_string(),
                        self.image_size[0].to_string(),
                        self.image_size[1].to_string(),
                        wp.pred_class_id.to_string(),
                        wp.pred_label.clone(),
                        wp.is_countable.to_string(),
                        format!("{:.6}", wp.probs[0]),
                        format!("{:.6}", wp.probs[1]),
                        format!("{:.6}", wp.probs[2]),
                    ])?;
                }
                for sec in &rep.sections {
                    let bx =
                        sec.region_bbox
                            .unwrap_or([0, 0, self.image_size[0], self.image_size[1]]);
                    wtr.write_record([
                        "section".to_string(),
                        bx[0].to_string(),
                        bx[1].to_string(),
                        bx[2].to_string(),
                        bx[3].to_string(),
                        sec.pred_class_id.to_string(),
                        sec.pred_label.clone(),
                        sec.is_countable.to_string(),
                        format!("{:.6}", sec.probs[0]),
                        format!("{:.6}", sec.probs[1]),
                        format!("{:.6}", sec.probs[2]),
                    ])?;
                }
            }
        } else {
            wtr.write_record([
                "instance_id",
                "label",
                "class_id",
                "score",
                "cx",
                "cy",
                "radius_px",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ])?;
            for (idx, det) in self.detections.iter().enumerate() {
                wtr.write_record([
                    (idx + 1).to_string(),
                    det.label.clone(),
                    det.class_id.to_string(),
                    format!("{:.6}", det.score),
                    format!("{:.3}", det.center_xy[0]),
                    format!("{:.3}", det.center_xy[1]),
                    format!("{:.3}", det.radius_px),
                    format!("{:.3}", det.bbox[0]),
                    format!("{:.3}", det.bbox[1]),
                    format!("{:.3}", det.bbox[2]),
                    format!("{:.3}", det.bbox[3]),
                ])?;
            }
        }
        wtr.flush()?;
        Ok(dest)
    }
}

/// High-resolution sliced ONNX inference engine orchestrating CenterNet, StarDist, and Countability models.
pub struct AutoCFUEngine {
    /// Validated configuration driving slicing and post-processing.
    pub config: EngineConfig,
    /// Loaded ONNX model runner.
    pub model: AutoDetectedModel,
    /// 2D overlapping tile grid slicer.
    pub slicer: PlateSlicer,
}

impl AutoCFUEngine {
    /// Load an `.onnx` model (with optional `auto` model-type detection) and initialize the engine.
    ///
    /// # Errors
    /// Returns configuration, ONNX loading, or slicer errors.
    pub fn new(model_path: impl AsRef<Path>, config: EngineConfig) -> Result<Self> {
        config.validate()?;
        let model = AutoDetectedModel::load(model_path, &config.inference)?;
        let slicer = PlateSlicer::from_config(&config.slicing)?;
        Ok(Self {
            config,
            model,
            slicer,
        })
    }

    /// Run inference on an image file loaded from disk.
    ///
    /// # Errors
    /// Returns [`AutoCFUError::FileNotFound`], [`AutoCFUError::Image`], or inference errors.
    pub fn predict_file(&mut self, image_path: impl AsRef<Path>) -> Result<InferenceResult> {
        let path = image_path.as_ref();
        if !path.exists() {
            return Err(AutoCFUError::FileNotFound(path.to_path_buf()));
        }
        let dyn_img = image::open(path)?;
        self.predict_image(&dyn_img, Some(path.to_string_lossy().into_owned()))
    }

    /// Run inference on an already-decoded [`DynamicImage`].
    ///
    /// # Errors
    /// Propagates ONNX inference, slicing, or merging errors.
    pub fn predict_image(
        &mut self,
        image: &DynamicImage,
        image_path: Option<String>,
    ) -> Result<InferenceResult> {
        let t0 = Instant::now();
        let width = image.width() as usize;
        let height = image.height() as usize;
        let model_type = self.model.model_type();

        let mut res = match &mut self.model {
            AutoDetectedModel::Countability(runner) => {
                let scope = self.config.inference.countability_scope;
                let thresh = self.config.inference.uncountable_threshold;
                let mut whole_plate = None;
                let mut sections = Vec::new();
                let mut num_slices = 0usize;

                if scope == CountabilityScope::Both || scope == CountabilityScope::Plate {
                    whole_plate =
                        Some(runner.predict(image, thresh, Some([0, 0, width, height]))?);
                    num_slices += 1;
                }
                if scope == CountabilityScope::Both || scope == CountabilityScope::Sections {
                    let windows = self.slicer.get_slice_windows(width, height)?;
                    for win in &windows {
                        #[allow(clippy::cast_possible_truncation)]
                        let crop = image.crop_imm(
                            win.xmin as u32,
                            win.ymin as u32,
                            win.width() as u32,
                            win.height() as u32,
                        );
                        let pred = runner.predict(
                            &crop,
                            thresh,
                            Some([win.xmin, win.ymin, win.xmax, win.ymax]),
                        )?;
                        sections.push(pred);
                    }
                    num_slices += windows.len();
                }

                let report = PlateCountabilityReport::from_parts(whole_plate, sections, thresh);
                InferenceResult {
                    image_path,
                    image_size: [width, height],
                    model_type,
                    colony_count: 0,
                    num_slices: num_slices.max(1),
                    inference_time_ms: 0.0,
                    countability: Some(report),
                    detections: Vec::new(),
                }
            }
            AutoDetectedModel::Centernet(_) | AutoDetectedModel::Stardist(_) => {
                let inf = &self.config.inference;
                let sli = &self.config.slicing;
                let (global_dets, num_slices) =
                    if width <= sli.slice_width && height <= sli.slice_height {
                        let tile_dets = match &mut self.model {
                            AutoDetectedModel::Centernet(r) => r.predict(image, inf)?,
                            AutoDetectedModel::Stardist(r) => r.predict(image, inf)?,
                            AutoDetectedModel::Countability(_) => Vec::new(),
                        };
                        (tile_dets, 1usize)
                    } else {
                        let windows = self.slicer.get_slice_windows(width, height)?;
                        let mut collected = Vec::new();
                        for win in &windows {
                            #[allow(clippy::cast_possible_truncation)]
                            let crop = image.crop_imm(
                                win.xmin as u32,
                                win.ymin as u32,
                                win.width() as u32,
                                win.height() as u32,
                            );
                            let tile_dets = match &mut self.model {
                                AutoDetectedModel::Centernet(r) => r.predict(&crop, inf)?,
                                AutoDetectedModel::Stardist(r) => r.predict(&crop, inf)?,
                                AutoDetectedModel::Countability(_) => Vec::new(),
                            };
                            for det in tile_dets {
                                collected.push(det.translated(win.xmin as f64, win.ymin as f64));
                            }
                        }
                        (collected, windows.len())
                    };

                let merged = apply_merger(
                    inf.merger_type,
                    inf.postprocess_match_threshold,
                    inf.match_metric,
                    &global_dets,
                )?;

                InferenceResult {
                    image_path,
                    image_size: [width, height],
                    model_type,
                    colony_count: merged.len(),
                    num_slices,
                    inference_time_ms: 0.0,
                    countability: None,
                    detections: merged,
                }
            }
        };

        res.inference_time_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok(res)
    }
}

/// Export an image-level summary CSV table across multiple inference results.
///
/// # Errors
/// Returns [`AutoCFUError::Io`] or [`AutoCFUError::Csv`] on write failure.
pub fn export_batch_summary_csv(
    results: &[InferenceResult],
    output_path: impl AsRef<Path>,
) -> Result<PathBuf> {
    let dest = output_path.as_ref().to_path_buf();
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut wtr = csv::Writer::from_path(&dest)?;
    wtr.write_record([
        "image_path",
        "model_type",
        "width",
        "height",
        "colony_count",
        "mean_radius_px",
        "is_countable",
        "pred_label",
        "num_slices",
        "inference_time_ms",
    ])?;

    for res in results {
        let is_cnt = res
            .countability
            .as_ref()
            .map_or_else(String::new, |rep| rep.is_countable.to_string());
        let pred_label = res
            .countability
            .as_ref()
            .and_then(|rep| rep.whole_plate.as_ref())
            .map_or_else(String::new, |wp| wp.pred_label.clone());

        wtr.write_record([
            res.image_path.clone().unwrap_or_default(),
            res.model_type.as_str().to_string(),
            res.image_size[0].to_string(),
            res.image_size[1].to_string(),
            res.colony_count.to_string(),
            format!("{:.3}", res.mean_radius_px()),
            is_cnt,
            pred_label,
            res.num_slices.to_string(),
            format!("{:.2}", res.inference_time_ms),
        ])?;
    }
    wtr.flush()?;
    Ok(dest)
}
