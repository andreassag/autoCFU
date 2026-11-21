#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

//! AutoCFU Standalone Multi-Model ONNX Inference Engine (Rust).
//!
//! Provides high-resolution sliced Petri dish inference with strict mathematical parity
//! against the Python `autocfu` package for:
//! - Radial CenterNet (`centernet`)
//! - 2D StarDist 32-ray star-convex instance segmentation (`stardist`)
//! - Plate & Section Countability Classification (`countability`)

pub mod config;
pub mod error;
pub mod merger;
pub mod model;
pub mod pipeline;
pub mod slicer;

pub use config::{
    CountabilityScope, DevicePreference, EngineConfig, InferenceConfig, MatchMetric, MergerType,
    ModelType, SlicingConfig,
};
pub use error::{AutoCFUError, Result};
pub use merger::{
    apply_merger, calculate_center_distance_ratio, calculate_ios, calculate_iou, detections_match,
    reconstruct_star_polygon, Detection, GreedyNMM, NMS,
};
pub use model::{
    detect_onnx_model_type, AutoDetectedModel, CenterNetOnnxModel, CountabilityOnnxModel,
    CountabilityResult, StarDistOnnxModel, COUNTABILITY_LABELS,
};
pub use pipeline::{
    export_batch_summary_csv, AutoCFUEngine, InferenceResult, PlateCountabilityReport,
    SectionSummary, IMAGE_EXTENSIONS,
};
pub use slicer::{PlateSlicer, SliceWindow};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slicer_edge_clamping_parity() {
        let slicer = PlateSlicer::new(640, 640, 0.20, 0.20).expect("valid slicer");
        assert_eq!(slicer.step_x, 512);
        assert_eq!(slicer.step_y, 512);

        let windows = slicer.get_slice_windows(1000, 1000).expect("valid windows");
        assert_eq!(
            windows,
            vec![
                SliceWindow {
                    xmin: 0,
                    ymin: 0,
                    xmax: 640,
                    ymax: 640
                },
                SliceWindow {
                    xmin: 360,
                    ymin: 0,
                    xmax: 1000,
                    ymax: 640
                },
                SliceWindow {
                    xmin: 0,
                    ymin: 360,
                    xmax: 640,
                    ymax: 1000
                },
                SliceWindow {
                    xmin: 360,
                    ymin: 360,
                    xmax: 1000,
                    ymax: 1000
                },
            ]
        );
    }

    #[test]
    fn test_greedy_nmm_center_dist_and_stardist_rays() {
        let rays_a = vec![10.0_f64; 32];
        let rays_b = vec![20.0_f64; 32];
        let det_a = Detection::from_stardist([100.0, 200.0], rays_a, None, 0.9, 0, "colony")
            .expect("valid stardist det");
        let det_b = Detection::from_stardist([103.0, 204.0], rays_b, None, 0.3, 0, "colony")
            .expect("valid stardist det");

        let ratio = calculate_center_distance_ratio(&det_a, &det_b);
        assert!((ratio - 0.5).abs() < 1e-6);

        let merger = GreedyNMM::new(0.6, MatchMetric::CenterDist);
        let merged = merger.merge(&[det_a, det_b]).expect("merge succeeds");
        assert_eq!(merged.len(), 1);
        assert!((merged[0].center_xy[0] - 100.75).abs() < 1e-6);
        assert!((merged[0].center_xy[1] - 201.0).abs() < 1e-6);
        assert!((merged[0].radius_px - 12.5).abs() < 1e-6);
        let poly = merged[0].polygon.as_ref().expect("polygon present");
        assert_eq!(poly.len(), 32);
        // k = 0 => cos(0) = 1, sin(0) = 0
        assert!((poly[0][0] - (100.75 + 12.5)).abs() < 1e-6);
        assert!((poly[0][1] - 201.0).abs() < 1e-6);
    }

    #[test]
    fn test_model_auto_detection_rules() {
        let sd = detect_onnx_model_type(
            &[
                "boxes".to_string(),
                "scores".to_string(),
                "labels".to_string(),
                "centers".to_string(),
                "rays".to_string(),
                "polygons".to_string(),
            ],
            None,
        )
        .expect("detect stardist");
        assert_eq!(sd, ModelType::Stardist);

        let cn = detect_onnx_model_type(
            &[
                "boxes".to_string(),
                "scores".to_string(),
                "labels".to_string(),
                "centers".to_string(),
                "radii".to_string(),
            ],
            None,
        )
        .expect("detect centernet");
        assert_eq!(cn, ModelType::Centernet);

        let cnt = detect_onnx_model_type(
            &[
                "logits".to_string(),
                "probs".to_string(),
                "pred_class".to_string(),
                "is_countable".to_string(),
            ],
            None,
        )
        .expect("detect countability");
        assert_eq!(cnt, ModelType::Countability);
    }

    #[test]
    fn test_uint16_mask_rasterization_and_tiff_export() {
        let dir = std::env::temp_dir().join("autocfu_rust_mask_test");
        let _ = std::fs::remove_dir_all(&dir);
        let det1 = Detection::from_center_radius([20.0, 20.0], 6.0, 0.95, 0, "colony");
        let det2 = Detection::from_stardist([60.0, 60.0], vec![8.0; 16], None, 0.85, 0, "colony")
            .expect("valid stardist");
        let res = InferenceResult {
            image_path: None,
            image_size: [100, 100],
            model_type: ModelType::Stardist,
            colony_count: 2,
            num_slices: 1,
            inference_time_ms: 1.0,
            countability: None,
            detections: vec![det1, det2],
        };
        let mask = res.to_mask_u16();
        assert_eq!(mask.len(), 100 * 100);
        assert_eq!(mask[20 * 100 + 20], 1);
        assert_eq!(mask[60 * 100 + 60], 2);
        assert_eq!(mask[0], 0);

        let tif_path = dir.join("mask.tif");
        res.save_mask(&tif_path).expect("save tiff");
        assert!(tif_path.exists());
        let png_path = dir.join("mask.png");
        res.save_mask(&png_path).expect("save png");
        assert!(png_path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_inference_config_validation_parity() {
        assert!(DevicePreference::parse("auto").is_ok());
        assert!(DevicePreference::parse("cpu").is_ok());
        assert!(DevicePreference::parse("cuda").is_ok());
        assert!(DevicePreference::parse("tpu").is_err());

        let mut cfg = InferenceConfig::default();
        assert!(cfg.validate().is_ok());

        cfg.intra_tile_nms_threshold = 0.0;
        assert!(cfg.validate().is_ok());
        cfg.intra_tile_nms_threshold = 2.0;
        assert!(cfg.validate().is_ok());

        cfg.intra_tile_nms_threshold = -0.1;
        assert!(cfg.validate().is_err());
        cfg.intra_tile_nms_threshold = 2.5;
        assert!(cfg.validate().is_err());
    }
}
