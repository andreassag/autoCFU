//! ONNX Runtime model runners for CenterNet, 2D StarDist, and Countability inference.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use image::{imageops::FilterType, DynamicImage};
use ndarray::{Array2, Array4, ArrayViewD};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::{TensorRef, Value, ValueType};
use serde::Serialize;

use crate::config::{DevicePreference, InferenceConfig, MatchMetric, MergerType, ModelType};
use crate::error::{AutoCFUError, Result};
use crate::merger::{apply_merger, Detection, NMS};

/// Canonical 3-class taxonomy emitted by the Countability classifier.
pub const COUNTABILITY_LABELS: [&str; 3] = ["empty", "countable", "uncountable"];

/// Prediction returned by a Countability classifier ONNX model.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CountabilityResult {
    /// Class probabilities `(p_empty, p_countable, p_uncountable)`.
    pub probs: [f64; 3],
    /// Raw model logits `(l_empty, l_countable, l_uncountable)`.
    pub logits: [f64; 3],
    /// Predicted argmax class index in `{0: empty, 1: countable, 2: uncountable}`.
    pub pred_class_id: usize,
    /// Predicted class label string.
    pub pred_label: String,
    /// Binary countability flag (`p_uncountable < uncountable_threshold`).
    pub is_countable: bool,
    /// Combined probability of `empty` or `countable`.
    pub countable_probability: f64,
    /// Probability of class `uncountable`.
    pub uncountable_probability: f64,
    /// Optional `[xmin, ymin, xmax, ymax]` pixel bounds for section predictions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region_bbox: Option<[usize; 4]>,
}

/// Detect the model family (`Centernet`, `Stardist`, or `Countability`) from metadata and outputs.
///
/// # Errors
/// Returns [`AutoCFUError::UnsupportedModel`] if outputs do not identify any supported architecture.
pub fn detect_onnx_model_type(
    output_names: &[String],
    metadata_model_type: Option<&str>,
) -> Result<ModelType> {
    if let Some(meta_val) = metadata_model_type {
        if let Ok(parsed) = ModelType::parse(meta_val) {
            if parsed != ModelType::Auto {
                return Ok(parsed);
            }
        }
    }

    let lowered: HashSet<String> = output_names
        .iter()
        .map(|s| s.trim().to_ascii_lowercase())
        .collect();

    if ["rays", "polygons", "ray_dists", "prob"]
        .iter()
        .any(|k| lowered.contains(*k))
    {
        return Ok(ModelType::Stardist);
    }
    if ["radii", "centers", "heatmap", "hm", "radius"]
        .iter()
        .any(|k| lowered.contains(*k))
    {
        return Ok(ModelType::Centernet);
    }
    if ["logits", "probs", "pred_class", "is_countable"]
        .iter()
        .any(|k| lowered.contains(*k))
    {
        return Ok(ModelType::Countability);
    }

    Err(AutoCFUError::UnsupportedModel(format!(
        "Cannot auto-detect AutoCFU model type from outputs {output_names:?}"
    )))
}

/// Resize an image to `input_size x input_size` and convert to NCHW float32 in `[0.0, 1.0]`.
#[must_use]
pub fn preprocess_rgb_image(image: &DynamicImage, input_size: usize) -> (Array4<f32>, (u32, u32)) {
    let rgb = image.to_rgb8();
    let orig_w = rgb.width();
    let orig_h = rgb.height();

    let proc_rgb = if orig_w as usize != input_size || orig_h as usize != input_size {
        #[allow(clippy::cast_possible_truncation)]
        image::imageops::resize(
            &rgb,
            input_size as u32,
            input_size as u32,
            FilterType::Triangle,
        )
    } else {
        rgb
    };

    let hw = input_size * input_size;
    let mut buffer = vec![0.0_f32; 3 * hw];
    let raw = proc_rgb.as_raw();
    for (idx, chunk) in raw.as_chunks::<3>().0.iter().enumerate() {
        buffer[idx] = f32::from(chunk[0]) / 255.0;
        buffer[hw + idx] = f32::from(chunk[1]) / 255.0;
        buffer[2 * hw + idx] = f32::from(chunk[2]) / 255.0;
    }

    // Shape 1 x 3 x input_size x input_size matches `buffer.len() == 3 * hw` exactly.
    let tensor = Array4::from_shape_vec((1, 3, input_size, input_size), buffer)
        .unwrap_or_else(|_| Array4::zeros((1, 3, input_size, input_size)));
    (tensor, (orig_w, orig_h))
}

fn find_output_index(output_names: &[String], candidates: &[&str]) -> Option<usize> {
    for &cand in candidates {
        if let Some(pos) = output_names.iter().position(|name| name == cand) {
            return Some(pos);
        }
    }
    let cand_lower: Vec<String> = candidates.iter().map(|c| c.to_ascii_lowercase()).collect();
    for low in cand_lower {
        if let Some(pos) = output_names
            .iter()
            .position(|name| name.to_ascii_lowercase() == low)
        {
            return Some(pos);
        }
    }
    None
}

fn extract_f64_flat(val: &Value) -> Result<(Vec<usize>, Vec<f64>)> {
    if let Ok(arr) = val.try_extract_array::<f32>() {
        let shape = arr.shape().to_vec();
        let data = arr.iter().map(|&v| f64::from(v)).collect();
        return Ok((shape, data));
    }
    if let Ok(arr) = val.try_extract_array::<f64>() {
        let shape = arr.shape().to_vec();
        let data = arr.iter().copied().collect();
        return Ok((shape, data));
    }
    Err(AutoCFUError::OnnxRuntime(
        "Failed to extract floating-point tensor from ONNX output".to_string(),
    ))
}

fn extract_i64_flat(val: &Value) -> Result<Vec<i64>> {
    if let Ok(arr) = val.try_extract_array::<i64>() {
        return Ok(arr.iter().copied().collect());
    }
    if let Ok(arr) = val.try_extract_array::<i32>() {
        return Ok(arr.iter().map(|&v| i64::from(v)).collect());
    }
    if let Ok(arr) = val.try_extract_array::<f32>() {
        #[allow(clippy::cast_possible_truncation)]
        return Ok(arr.iter().map(|&v| v as i64).collect());
    }
    Err(AutoCFUError::OnnxRuntime(
        "Failed to extract integer label tensor from ONNX output".to_string(),
    ))
}

fn max_pool_3x3(hm: &Array2<f64>) -> Array2<f64> {
    let (rows, cols) = hm.dim();
    let mut pooled = Array2::from_elem((rows, cols), f64::NEG_INFINITY);
    for y in 0..rows {
        let y0 = y.saturating_sub(1);
        let y1 = (y + 1).min(rows.saturating_sub(1));
        for x in 0..cols {
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(cols.saturating_sub(1));
            let mut best = f64::NEG_INFINITY;
            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    let val = hm[(ny, nx)];
                    if val > best {
                        best = val;
                    }
                }
            }
            pooled[(y, x)] = best;
        }
    }
    pooled
}

fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-60.0, 60.0)).exp())
}

fn softmax_3(logits: [f64; 3]) -> [f64; 3] {
    let max_v = logits[0].max(logits[1]).max(logits[2]);
    let exps = [
        (logits[0] - max_v).exp(),
        (logits[1] - max_v).exp(),
        (logits[2] - max_v).exp(),
    ];
    let sum = exps[0] + exps[1] + exps[2];
    if sum <= 0.0 {
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    } else {
        [exps[0] / sum, exps[1] / sum, exps[2] / sum]
    }
}

/// Process-wide ONNX Runtime environment singleton initialized on first model load.
static SHARED_ORT_ENV: Mutex<Option<Environment>> = Mutex::new(None);

fn get_or_init_environment(device: DevicePreference) -> Result<Environment> {
    let mut guard = SHARED_ORT_ENV.lock().map_err(|_| {
        AutoCFUError::OnnxRuntime("ONNX Runtime environment mutex poisoned".to_string())
    })?;
    if let Some(env) = guard.as_ref() {
        return Ok(env.clone());
    }

    let env = match device {
        DevicePreference::Auto => {
            #[cfg(feature = "cuda")]
            {
                ort::init()
                    .with_name("autocfu")
                    .with_execution_providers([ort::ep::CUDA::default().build()])
                    .build()?
            }
            #[cfg(not(feature = "cuda"))]
            {
                ort::init().with_name("autocfu").build()?
            }
        }
        DevicePreference::Cpu => ort::init().with_name("autocfu").build()?,
        DevicePreference::Cuda => {
            #[cfg(feature = "cuda")]
            {
                ort::init()
                    .with_name("autocfu")
                    .with_execution_providers([ort::ep::CUDA::default().build()])
                    .build()?
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(AutoCFUError::InvalidConfig(
                    "Requested device='cuda', but this binary was compiled without the 'cuda' feature".to_string(),
                ));
            }
        }
    };

    *guard = Some(env.clone());
    Ok(env)
}

fn build_session(
    model_path: &Path,
    device: DevicePreference,
) -> Result<(Environment, Session, usize)> {
    if !model_path.exists() {
        return Err(AutoCFUError::FileNotFound(model_path.to_path_buf()));
    }

    let env = get_or_init_environment(device)?;
    let session = Session::builder(&env)?.commit_from_file(model_path)?;
    let mut inferred_size = 640usize;
    if let Some(first_input) = session.inputs().first() {
        if let ValueType::Tensor { shape, .. } = first_input.dtype() {
            if shape.len() == 4 && shape[2] > 0 {
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                {
                    inferred_size = shape[2] as usize;
                }
            }
        }
    }
    Ok((env, session, inferred_size))
}

fn prepare_inputs<'a>(
    input_names: &'a [String],
    img_tensor: TensorRef<'a, f32>,
    orig_sizes: &'a Array2<i64>,
) -> Result<Vec<(&'a str, ort::session::SessionInputValue<'a>)>> {
    let primary_name = input_names.first().map_or("images", String::as_str);
    let mut inputs: Vec<(&'a str, ort::session::SessionInputValue<'a>)> =
        vec![(primary_name, img_tensor.into())];
    if input_names.iter().any(|n| n == "orig_target_sizes") {
        let size_tensor = TensorRef::from_array_view(orig_sizes.view())?;
        inputs.push(("orig_target_sizes", size_tensor.into()));
    }
    Ok(inputs)
}

fn filter_and_suppress(
    detections: Vec<Detection>,
    config: &InferenceConfig,
) -> Result<Vec<Detection>> {
    let mut filtered = Vec::new();
    for det in detections {
        if !det.score.is_finite() || det.score < config.confidence_threshold {
            continue;
        }
        if let Some(target_cls) = config.target_class_id {
            if det.class_id != target_cls {
                continue;
            }
        }
        if det.radius_px < config.min_radius || det.radius_px > config.max_radius {
            continue;
        }
        let w = det.width();
        let h = det.height();
        if w < config.min_box_size || h < config.min_box_size {
            continue;
        }
        if w > config.max_box_size || h > config.max_box_size {
            continue;
        }
        let min_side = w.min(h);
        if min_side <= 0.0 {
            continue;
        }
        let aspect = (w / min_side).max(h / min_side);
        if aspect > config.max_aspect_ratio {
            continue;
        }
        filtered.push(det);
    }

    if filtered.is_empty() || config.intra_tile_nms_threshold <= 0.0 {
        return Ok(filtered);
    }

    if config.intra_tile_nms_threshold < 1.0 {
        return Ok(NMS::new(config.intra_tile_nms_threshold, MatchMetric::Ios).merge(&filtered));
    }
    apply_merger(
        MergerType::None,
        config.intra_tile_nms_threshold,
        MatchMetric::Ios,
        &filtered,
    )
}

fn extract_dense_f64(arr: &ArrayViewD<'_, f32>) -> (usize, usize, usize, Vec<f64>) {
    let shape = arr.shape();
    let (c, h, w) = match shape {
        [_, c, h, w] => (*c, *h, *w),
        [c, h, w] => (*c, *h, *w),
        [h, w] => (1, *h, *w),
        _ => (0, 0, 0),
    };
    let data = arr.iter().map(|&v| f64::from(v)).collect();
    (c, h, w, data)
}

/// ONNX Runtime runner for Radial CenterNet models.
pub struct CenterNetOnnxModel {
    _env: Environment,
    session: Session,
    input_names: Vec<String>,
    output_names: Vec<String>,
    /// Input square spatial size expected by the ONNX graph.
    pub input_size: usize,
    /// Top-K raw peaks retained during raw-map fallback decoding.
    pub top_k: usize,
}

impl CenterNetOnnxModel {
    /// Load a Radial CenterNet ONNX model from disk.
    ///
    /// # Errors
    /// Returns an error if loading or initializing the ONNX session fails.
    pub fn from_file(model_path: impl AsRef<Path>, config: &InferenceConfig) -> Result<Self> {
        let (_env, session, graph_size) = build_session(model_path.as_ref(), config.device)?;
        Ok(Self::from_session(_env, session, graph_size, config))
    }

    fn from_session(
        _env: Environment,
        session: Session,
        graph_size: usize,
        config: &InferenceConfig,
    ) -> Self {
        let input_names = session
            .inputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let output_names = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        Self {
            _env,
            session,
            input_names,
            output_names,
            input_size: if graph_size > 0 {
                graph_size
            } else {
                config.input_size
            },
            top_k: config.top_k,
        }
    }

    /// Run CenterNet inference on a single image or tile crop.
    ///
    /// # Errors
    /// Returns an error if ONNX execution or output tensor parsing fails.
    pub fn predict(
        &mut self,
        image: &DynamicImage,
        config: &InferenceConfig,
    ) -> Result<Vec<Detection>> {
        let (nchw, (orig_w, orig_h)) = preprocess_rgb_image(image, self.input_size);
        let scale_x = f64::from(orig_w) / (self.input_size as f64);
        let scale_y = f64::from(orig_h) / (self.input_size as f64);
        let scale_r = 0.5 * (scale_x + scale_y);

        let output_names = self.output_names.clone();
        let input_names = self.input_names.clone();
        let img_tensor = TensorRef::from_array_view(nchw.view())?;
        let orig_sizes =
            Array2::from_shape_vec((1, 2), vec![self.input_size as i64, self.input_size as i64])
                .map_err(|e| AutoCFUError::InvalidGeometry(e.to_string()))?;
        let inputs = prepare_inputs(&input_names, img_tensor, &orig_sizes)?;
        let outputs = self.session.run(inputs)?;

        let scores_idx = find_output_index(&output_names, &["scores", "pred_scores", "confs"]);
        let centers_idx = find_output_index(&output_names, &["centers", "pred_centers"]);
        let radii_idx = find_output_index(&output_names, &["radii", "pred_radii"]);
        let boxes_idx = find_output_index(&output_names, &["boxes", "pred_boxes", "bboxes"]);
        let labels_idx = find_output_index(&output_names, &["labels", "pred_labels", "classes"]);

        let raw_detections = if let Some(si) = scores_idx {
            if centers_idx.is_some() || boxes_idx.is_some() {
                let (_, scores) = extract_f64_flat(&outputs[si])?;
                let labels = match labels_idx {
                    Some(li) => extract_i64_flat(&outputs[li])?,
                    None => vec![0_i64; scores.len()],
                };
                let centers = match centers_idx {
                    Some(ci) => Some(extract_f64_flat(&outputs[ci])?.1),
                    None => None,
                };
                let radii = match radii_idx {
                    Some(ri) => Some(extract_f64_flat(&outputs[ri])?.1),
                    None => None,
                };
                let boxes = match boxes_idx {
                    Some(bi) => Some(extract_f64_flat(&outputs[bi])?.1),
                    None => None,
                };

                let mut list = Vec::new();
                for (i, &score) in scores.iter().enumerate() {
                    if score < config.confidence_threshold {
                        continue;
                    }
                    let cls_id = labels.get(i).copied().unwrap_or(0);
                    if let (Some(c_data), Some(r_data)) = (&centers, &radii) {
                        let cx = c_data.get(2 * i).copied().unwrap_or(0.0) * scale_x;
                        let cy = c_data.get(2 * i + 1).copied().unwrap_or(0.0) * scale_y;
                        let r = (r_data.get(i).copied().unwrap_or(0.0) * scale_r).max(0.0);
                        let mut det = Detection::from_center_radius(
                            [cx, cy],
                            r,
                            score,
                            cls_id,
                            config.class_name.clone(),
                        );
                        if let Some(b_data) = &boxes {
                            if b_data.len() >= 4 * (i + 1) {
                                det.bbox = [
                                    b_data[4 * i] * scale_x,
                                    b_data[4 * i + 1] * scale_y,
                                    b_data[4 * i + 2] * scale_x,
                                    b_data[4 * i + 3] * scale_y,
                                ];
                            }
                        }
                        list.push(det);
                    } else if let Some(b_data) = &boxes {
                        if b_data.len() >= 4 * (i + 1) {
                            list.push(Detection::from_bbox(
                                [
                                    b_data[4 * i] * scale_x,
                                    b_data[4 * i + 1] * scale_y,
                                    b_data[4 * i + 2] * scale_x,
                                    b_data[4 * i + 3] * scale_y,
                                ],
                                score,
                                cls_id,
                                config.class_name.clone(),
                            ));
                        }
                    }
                }
                list
            } else {
                Vec::new()
            }
        } else {
            let hm_idx = find_output_index(&output_names, &["heatmap", "hm"]);
            let off_idx = find_output_index(&output_names, &["offset", "off", "reg"]);
            let rad_idx = find_output_index(&output_names, &["radius", "rad"]);

            let (Some(hi), Some(oi), Some(ri)) = (hm_idx, off_idx, rad_idx) else {
                return Err(AutoCFUError::UnsupportedModel(format!(
                    "CenterNet outputs {output_names:?} match neither decoded tensors nor raw maps"
                )));
            };

            let hm_view = outputs[hi].try_extract_array::<f32>()?;
            let off_view = outputs[oi].try_extract_array::<f32>()?;
            let rad_view = outputs[ri].try_extract_array::<f32>()?;

            let (num_cls, fh, fw, hm_data) = extract_dense_f64(&hm_view);
            let (_, _, _, off_data) = extract_dense_f64(&off_view);
            let (_, _, _, rad_data) = extract_dense_f64(&rad_view);
            if fh == 0 || fw == 0 {
                return Err(AutoCFUError::UnsupportedModel(
                    "Invalid CenterNet raw heatmap tensor shape".to_string(),
                ));
            }
            let stride = (self.input_size as f64) / (fh as f64);
            let need_sigmoid = hm_data.iter().any(|&v| !(0.0..=1.0).contains(&v));

            let mut list = Vec::new();
            let hw = fh * fw;
            for cls_id in 0..num_cls {
                let slice_start = cls_id * hw;
                let mut hm_2d = Array2::zeros((fh, fw));
                for y in 0..fh {
                    for x in 0..fw {
                        let raw_v = hm_data[slice_start + y * fw + x];
                        hm_2d[(y, x)] = if need_sigmoid {
                            sigmoid_f64(raw_v)
                        } else {
                            raw_v
                        };
                    }
                }
                let pooled = max_pool_3x3(&hm_2d);
                let mut peaks: Vec<(f64, usize, usize)> = Vec::new();
                for y in 0..fh {
                    for x in 0..fw {
                        let s = hm_2d[(y, x)];
                        if (s - pooled[(y, x)]).abs() <= f64::EPSILON
                            && s >= config.confidence_threshold
                        {
                            peaks.push((s, y, x));
                        }
                    }
                }
                peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
                peaks.truncate(self.top_k);
                for (score, gy, gx) in peaks {
                    let lin = gy * fw + gx;
                    let dx = off_data.get(lin).copied().unwrap_or(0.0);
                    let dy = off_data.get(hw + lin).copied().unwrap_or(0.0);
                    let r_px = (rad_data.get(lin).copied().unwrap_or(0.0) * scale_r).max(0.0);
                    let cx = ((gx as f64) + dx) * stride * scale_x;
                    let cy = ((gy as f64) + dy) * stride * scale_y;
                    list.push(Detection::from_center_radius(
                        [cx, cy],
                        r_px,
                        score,
                        cls_id as i64,
                        config.class_name.clone(),
                    ));
                }
            }
            list.sort_by(|a, b| b.score.total_cmp(&a.score));
            list.truncate(self.top_k);
            list
        };

        filter_and_suppress(raw_detections, config)
    }
}

/// ONNX Runtime runner for 2D StarDist star-convex polygon models.
pub struct StarDistOnnxModel {
    _env: Environment,
    session: Session,
    input_names: Vec<String>,
    output_names: Vec<String>,
    /// Input square spatial size expected by the ONNX graph.
    pub input_size: usize,
    /// Top-K raw peaks retained during raw-map fallback decoding.
    pub top_k: usize,
}

impl StarDistOnnxModel {
    /// Load a 2D StarDist ONNX model from disk.
    ///
    /// # Errors
    /// Returns an error if loading or initializing the ONNX session fails.
    pub fn from_file(model_path: impl AsRef<Path>, config: &InferenceConfig) -> Result<Self> {
        let (_env, session, graph_size) = build_session(model_path.as_ref(), config.device)?;
        Ok(Self::from_session(_env, session, graph_size, config))
    }

    fn from_session(
        _env: Environment,
        session: Session,
        graph_size: usize,
        config: &InferenceConfig,
    ) -> Self {
        let input_names = session
            .inputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let output_names = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        Self {
            _env,
            session,
            input_names,
            output_names,
            input_size: if graph_size > 0 {
                graph_size
            } else {
                config.input_size
            },
            top_k: config.top_k,
        }
    }

    /// Run 2D StarDist inference on a single image or tile crop.
    ///
    /// # Errors
    /// Returns an error if ONNX execution, polygon decoding, or geometry filtering fails.
    pub fn predict(
        &mut self,
        image: &DynamicImage,
        config: &InferenceConfig,
    ) -> Result<Vec<Detection>> {
        let (nchw, (orig_w, orig_h)) = preprocess_rgb_image(image, self.input_size);
        let scale_x = f64::from(orig_w) / (self.input_size as f64);
        let scale_y = f64::from(orig_h) / (self.input_size as f64);
        let scale_r = 0.5 * (scale_x + scale_y);

        let output_names = self.output_names.clone();
        let input_names = self.input_names.clone();
        let img_tensor = TensorRef::from_array_view(nchw.view())?;
        let orig_sizes =
            Array2::from_shape_vec((1, 2), vec![self.input_size as i64, self.input_size as i64])
                .map_err(|e| AutoCFUError::InvalidGeometry(e.to_string()))?;
        let inputs = prepare_inputs(&input_names, img_tensor, &orig_sizes)?;
        let outputs = self.session.run(inputs)?;

        let scores_idx = find_output_index(&output_names, &["scores", "pred_scores", "confs"]);
        let centers_idx = find_output_index(&output_names, &["centers", "pred_centers"]);
        let rays_idx = find_output_index(&output_names, &["rays", "pred_rays", "ray_dists"]);
        let polys_idx = find_output_index(&output_names, &["polygons", "pred_polygons"]);
        let labels_idx = find_output_index(&output_names, &["labels", "pred_labels", "classes"]);

        let raw_detections = if let (Some(si), Some(ci)) = (scores_idx, centers_idx) {
            if rays_idx.is_none() && polys_idx.is_none() {
                return Err(AutoCFUError::UnsupportedModel(
                    "StarDist model outputs lack both 'rays' and 'polygons'".to_string(),
                ));
            }
            let (_, scores) = extract_f64_flat(&outputs[si])?;
            let (_, centers) = extract_f64_flat(&outputs[ci])?;
            let labels = match labels_idx {
                Some(li) => extract_i64_flat(&outputs[li])?,
                None => vec![0_i64; scores.len()],
            };
            let rays_info = match rays_idx {
                Some(ri) => Some(extract_f64_flat(&outputs[ri])?),
                None => None,
            };
            let polys_info = match polys_idx {
                Some(pi) => Some(extract_f64_flat(&outputs[pi])?),
                None => None,
            };

            let num_rays = rays_info
                .as_ref()
                .and_then(|(shape, _)| shape.last().copied())
                .or_else(|| {
                    polys_info
                        .as_ref()
                        .and_then(|(shape, _)| shape.get(shape.len().saturating_sub(2)).copied())
                })
                .unwrap_or(config.num_rays);

            let mut list = Vec::new();
            for (i, &score) in scores.iter().enumerate() {
                if score < config.confidence_threshold {
                    continue;
                }
                let cls_id = labels.get(i).copied().unwrap_or(0);
                let cx = centers.get(2 * i).copied().unwrap_or(0.0) * scale_x;
                let cy = centers.get(2 * i + 1).copied().unwrap_or(0.0) * scale_y;

                let ray_vec: Vec<f64> = if let Some((_, r_data)) = &rays_info {
                    let start = i * num_rays;
                    let end = (start + num_rays).min(r_data.len());
                    r_data[start..end]
                        .iter()
                        .map(|&v| (v * scale_r).max(0.0))
                        .collect()
                } else {
                    Vec::new()
                };

                let poly_opt: Option<Vec<[f64; 2]>> = polys_info.as_ref().map(|(_, p_data)| {
                    let start = i * num_rays * 2;
                    let mut pts = Vec::with_capacity(num_rays);
                    for k in 0..num_rays {
                        let px = p_data.get(start + 2 * k).copied().unwrap_or(0.0) * scale_x;
                        let py = p_data.get(start + 2 * k + 1).copied().unwrap_or(0.0) * scale_y;
                        pts.push([px, py]);
                    }
                    pts
                });

                let det = if !ray_vec.is_empty() {
                    Detection::from_stardist(
                        [cx, cy],
                        ray_vec,
                        poly_opt,
                        score,
                        cls_id,
                        config.class_name.clone(),
                    )?
                } else if let Some(pts) = poly_opt {
                    let mut dummy_rays = Vec::with_capacity(pts.len());
                    for pt in &pts {
                        dummy_rays.push((pt[0] - cx).hypot(pt[1] - cy));
                    }
                    Detection::from_stardist(
                        [cx, cy],
                        dummy_rays,
                        Some(pts),
                        score,
                        cls_id,
                        config.class_name.clone(),
                    )?
                } else {
                    continue;
                };
                list.push(det);
            }
            list
        } else {
            let prob_idx = find_output_index(&output_names, &["prob", "probs", "object_prob"]);
            let raw_rays_idx = find_output_index(&output_names, &["rays", "ray_dists", "dist"]);
            let (Some(pi), Some(ri)) = (prob_idx, raw_rays_idx) else {
                return Err(AutoCFUError::UnsupportedModel(format!(
                    "StarDist outputs {output_names:?} match neither decoded tensors nor raw maps"
                )));
            };

            let prob_view = outputs[pi].try_extract_array::<f32>()?;
            let ray_view = outputs[ri].try_extract_array::<f32>()?;
            let (_, fh, fw, prob_data) = extract_dense_f64(&prob_view);
            let (num_rays, rf_h, rf_w, ray_data) = extract_dense_f64(&ray_view);
            if fh == 0 || fw == 0 || rf_h != fh || rf_w != fw || num_rays == 0 {
                return Err(AutoCFUError::UnsupportedModel(
                    "Invalid StarDist raw probability or ray map dimensions".to_string(),
                ));
            }
            let stride = (self.input_size as f64) / (fh as f64);
            let need_sigmoid = prob_data.iter().any(|&v| !(0.0..=1.0).contains(&v));
            let mut prob_2d = Array2::zeros((fh, fw));
            for y in 0..fh {
                for x in 0..fw {
                    let v = prob_data[y * fw + x];
                    prob_2d[(y, x)] = if need_sigmoid { sigmoid_f64(v) } else { v };
                }
            }
            let pooled = max_pool_3x3(&prob_2d);
            let mut peaks = Vec::new();
            for y in 0..fh {
                for x in 0..fw {
                    let s = prob_2d[(y, x)];
                    if (s - pooled[(y, x)]).abs() <= f64::EPSILON
                        && s >= config.confidence_threshold
                    {
                        peaks.push((s, y, x));
                    }
                }
            }
            peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
            peaks.truncate(self.top_k);

            let hw = fh * fw;
            let mut list = Vec::with_capacity(peaks.len());
            for (score, gy, gx) in peaks {
                let cx = ((gx as f64) + 0.5) * stride * scale_x;
                let cy = ((gy as f64) + 0.5) * stride * scale_y;
                let mut r_vec = Vec::with_capacity(num_rays);
                for k in 0..num_rays {
                    let val = ray_data.get(k * hw + gy * fw + gx).copied().unwrap_or(0.0);
                    r_vec.push((val * scale_r).max(0.0));
                }
                list.push(Detection::from_stardist(
                    [cx, cy],
                    r_vec,
                    None,
                    score,
                    0,
                    config.class_name.clone(),
                )?);
            }
            list
        };

        filter_and_suppress(raw_detections, config)
    }
}

/// ONNX Runtime runner for 3-class Plate & Section Countability classification models.
pub struct CountabilityOnnxModel {
    _env: Environment,
    session: Session,
    input_names: Vec<String>,
    output_names: Vec<String>,
    /// Input square spatial size expected by the ONNX graph.
    pub input_size: usize,
}

impl CountabilityOnnxModel {
    /// Load a Countability Classifier ONNX model from disk.
    ///
    /// # Errors
    /// Returns an error if loading or initializing the ONNX session fails.
    pub fn from_file(model_path: impl AsRef<Path>, config: &InferenceConfig) -> Result<Self> {
        let (_env, session, graph_size) = build_session(model_path.as_ref(), config.device)?;
        Ok(Self::from_session(_env, session, graph_size, config))
    }

    fn from_session(
        _env: Environment,
        session: Session,
        graph_size: usize,
        config: &InferenceConfig,
    ) -> Self {
        let input_names = session
            .inputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let output_names = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        Self {
            _env,
            session,
            input_names,
            output_names,
            input_size: if graph_size > 0 {
                graph_size
            } else {
                config.input_size
            },
        }
    }

    /// Classify an image or plate section into `empty`, `countable`, or `uncountable`.
    ///
    /// # Errors
    /// Returns an error if ONNX execution or output tensor parsing fails.
    pub fn predict(
        &mut self,
        image: &DynamicImage,
        uncountable_threshold: f64,
        region_bbox: Option<[usize; 4]>,
    ) -> Result<CountabilityResult> {
        let (nchw, _) = preprocess_rgb_image(image, self.input_size);
        let output_names = self.output_names.clone();
        let input_names = self.input_names.clone();
        let img_tensor = TensorRef::from_array_view(nchw.view())?;
        let orig_sizes =
            Array2::from_shape_vec((1, 2), vec![self.input_size as i64, self.input_size as i64])
                .map_err(|e| AutoCFUError::InvalidGeometry(e.to_string()))?;
        let inputs = prepare_inputs(&input_names, img_tensor, &orig_sizes)?;
        let outputs = self.session.run(inputs)?;

        let logits_idx = find_output_index(&output_names, &["logits"]);
        let probs_idx = find_output_index(&output_names, &["probs", "probabilities", "scores"]);

        if logits_idx.is_none() && probs_idx.is_none() {
            return Err(AutoCFUError::UnsupportedModel(format!(
                "Countability ONNX outputs {output_names:?} lack both 'probs' and 'logits'"
            )));
        }

        let probs_arr: [f64; 3];
        let logits_arr: [f64; 3];

        if let Some(pi) = probs_idx {
            let (_, p_vec) = extract_f64_flat(&outputs[pi])?;
            if p_vec.len() < 3 {
                return Err(AutoCFUError::UnsupportedModel(format!(
                    "Expected 3 countability probabilities, got {}",
                    p_vec.len()
                )));
            }
            probs_arr = [p_vec[0], p_vec[1], p_vec[2]];
            if let Some(li) = logits_idx {
                let (_, l_vec) = extract_f64_flat(&outputs[li])?;
                logits_arr = if l_vec.len() >= 3 {
                    [l_vec[0], l_vec[1], l_vec[2]]
                } else {
                    [
                        probs_arr[0].max(1e-7).ln(),
                        probs_arr[1].max(1e-7).ln(),
                        probs_arr[2].max(1e-7).ln(),
                    ]
                };
            } else {
                logits_arr = [
                    probs_arr[0].max(1e-7).ln(),
                    probs_arr[1].max(1e-7).ln(),
                    probs_arr[2].max(1e-7).ln(),
                ];
            }
        } else {
            let li = logits_idx.ok_or_else(|| {
                AutoCFUError::UnsupportedModel("Missing countability logits tensor".to_string())
            })?;
            let (_, l_vec) = extract_f64_flat(&outputs[li])?;
            if l_vec.len() < 3 {
                return Err(AutoCFUError::UnsupportedModel(format!(
                    "Expected 3 countability logits, got {}",
                    l_vec.len()
                )));
            }
            logits_arr = [l_vec[0], l_vec[1], l_vec[2]];
            probs_arr = softmax_3(logits_arr);
        }

        let mut pred_cls = 0usize;
        if probs_arr[1] > probs_arr[pred_cls] {
            pred_cls = 1;
        }
        if probs_arr[2] > probs_arr[pred_cls] {
            pred_cls = 2;
        }
        let pred_label = COUNTABILITY_LABELS[pred_cls].to_string();
        let countable_prob = probs_arr[0] + probs_arr[1];
        let uncountable_prob = probs_arr[2];
        let is_countable = uncountable_prob < uncountable_threshold;

        Ok(CountabilityResult {
            probs: probs_arr,
            logits: logits_arr,
            pred_class_id: pred_cls,
            pred_label,
            is_countable,
            countable_probability: countable_prob,
            uncountable_probability: uncountable_prob,
            region_bbox,
        })
    }
}

/// Polymorphic ONNX model runner wrapping all three supported AutoCFU architectures.
pub enum AutoDetectedModel {
    /// Radial CenterNet colony detector.
    Centernet(CenterNetOnnxModel),
    /// 2D StarDist star-convex polygon segmentation model.
    Stardist(StarDistOnnxModel),
    /// 3-class Plate & Section Countability classifier.
    Countability(CountabilityOnnxModel),
}

impl AutoDetectedModel {
    /// Inspect an `.onnx` file's outputs and metadata and load the corresponding model runner.
    ///
    /// # Errors
    /// Returns [`AutoCFUError::FileNotFound`], [`AutoCFUError::UnsupportedModel`], or ORT errors.
    pub fn load(model_path: impl AsRef<Path>, config: &InferenceConfig) -> Result<Self> {
        let path: PathBuf = model_path.as_ref().to_path_buf();
        let (env, session, graph_size) = build_session(&path, config.device)?;
        let resolved_type = if config.model_type == ModelType::Auto {
            let out_names: Vec<String> = session
                .outputs()
                .iter()
                .map(|o| o.name().to_string())
                .collect();
            let meta_type = session
                .metadata()
                .ok()
                .and_then(|m| m.custom("autocfu_model_type"));
            detect_onnx_model_type(&out_names, meta_type.as_deref())?
        } else {
            config.model_type
        };

        match resolved_type {
            ModelType::Centernet => Ok(Self::Centernet(CenterNetOnnxModel::from_session(
                env, session, graph_size, config,
            ))),
            ModelType::Stardist => Ok(Self::Stardist(StarDistOnnxModel::from_session(
                env, session, graph_size, config,
            ))),
            ModelType::Countability => Ok(Self::Countability(CountabilityOnnxModel::from_session(
                env, session, graph_size, config,
            ))),
            ModelType::Auto => Err(AutoCFUError::UnsupportedModel(
                "Model type remained Auto after inspection".to_string(),
            )),
        }
    }

    /// Return the resolved [`ModelType`] variant.
    #[must_use]
    pub const fn model_type(&self) -> ModelType {
        match self {
            Self::Centernet(_) => ModelType::Centernet,
            Self::Stardist(_) => ModelType::Stardist,
            Self::Countability(_) => ModelType::Countability,
        }
    }
}
