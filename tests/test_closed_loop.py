from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from src.raod_eras.config import DatasetConfig
from src.raod_eras.datasets import SMIYCRoadObstacleDataset
from src.raod_eras.datasets import Sample
from src.raod_eras.experiment import remove_small_components, select_samples
from src.raod_eras.metrics import PixelMetricAccumulator, evaluate_binary
from src.raod_eras.score_to_mask import PromptConfig, farthest_point_prompts, score_boxes
from src.raod_eras.object_refinement import AnomalyInstance, refine_objects
from src.raod_eras.risk_planning import plan_risk_response


class ClosedLoopTests(unittest.TestCase):
    def test_dataset_excludes_void_pixels(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "images"
            gt_dir = root / "gt"
            image_dir.mkdir()
            gt_dir.mkdir()
            Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(image_dir / "sample.png")
            Image.fromarray(np.asarray([[0, 1, 255], [0, 0, 255]], dtype=np.uint8)).save(gt_dir / "sample.png")
            dataset = SMIYCRoadObstacleDataset(
                DatasetConfig(
                    image_dir=image_dir,
                    gt_dir=gt_dir,
                    image_glob="*.png",
                    gt_suffix=".png",
                    positive_label=1,
                    ignore_label=255,
                )
            )
            loaded = dataset.load(dataset.list_samples()[0])
            self.assertEqual(int(loaded.gt.sum()), 1)
            self.assertEqual(int(loaded.valid.sum()), 4)
            self.assertFalse(bool(loaded.valid[0, 2]))

    def test_seeded_object_refinement_returns_instance(self) -> None:
        score = np.zeros((64, 64), dtype=np.float32)
        score[30:45, 27:39] = 0.48
        score[34:41, 30:36] = 0.90
        heatmap, mask, instances = refine_objects(score)
        self.assertEqual(heatmap.shape, score.shape)
        self.assertTrue(mask.any())
        self.assertGreaterEqual(len(instances), 1)
        self.assertGreater(instances[0].confidence, 0.0)
        self.assertTrue(np.all(heatmap[score == 0] == 0))
        self.assertGreater(float(heatmap[mask].mean()), float(score[mask].mean()))

    def test_perfect_binary_metrics(self) -> None:
        gt = np.zeros((48, 48), dtype=bool)
        gt[15:30, 17:32] = True
        metrics = evaluate_binary(gt.copy(), gt, np.ones_like(gt, dtype=bool))
        self.assertAlmostEqual(metrics["fixed_f1"], 1.0)
        self.assertAlmostEqual(metrics["fixed_iou"], 1.0)
        self.assertAlmostEqual(metrics["component_f1"], 1.0)
        self.assertAlmostEqual(metrics["boundary_f1"], 1.0)

    def test_risk_planner_returns_scored_candidates(self) -> None:
        heatmap = np.zeros((80, 120), dtype=np.float32)
        heatmap[50:70, 55:68] = 1.0
        instance = AnomalyInstance(
            instance_id=1,
            bbox_xyxy=(55, 50, 67, 69),
            area=260,
            mean_anomaly=0.9,
            boundary_contrast=0.5,
            road_overlap=1.0,
            lane_overlap=1.0,
            near_weight=0.9,
            confidence=0.9,
            uncertainty=0.1,
            distance_proxy=0.1,
        )
        plan = plan_risk_response(heatmap, [instance])
        self.assertIn(plan["selected_action"], {"keep_lane", "shift_left", "shift_right", "brake_or_stop"})
        self.assertGreaterEqual(len(plan["candidate_trajectories"]), 3)

    def test_small_component_cleanup(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[2:4, 2:4] = True
        mask[20:40, 20:40] = True
        cleaned = remove_small_components(mask, 0.001)
        self.assertFalse(cleaned[2:4, 2:4].any())
        self.assertTrue(cleaned[20:40, 20:40].all())

    def test_risk_prompt_fallback_for_large_region(self) -> None:
        score = np.zeros((100, 160), dtype=np.float32)
        score[45:95, 20:145] = 0.80
        score[70:76, 75:82] = 1.0
        boxes = score_boxes(score, PromptConfig(max_area_ratio=0.05), risk_aware=True)
        self.assertGreater(len(boxes), 0)

    def test_farthest_points_are_deterministic_and_road_constrained(self) -> None:
        score = np.zeros((100, 160), dtype=np.float32)
        score[10:20, 10:20] = 0.95
        score[55:95, 35:130] = 0.80
        config = PromptConfig(max_point_prompts=8, point_candidate_limit=200)
        first = farthest_point_prompts(score, config, road_aware=True)
        second = farthest_point_prompts(score, config, road_aware=True)
        self.assertEqual(first, second)
        self.assertGreater(len(first), 1)
        self.assertTrue(all(y >= 40 for _, y in first))
        self.assertGreater(len(set(first)), 1)

    def test_streaming_metrics_and_stratified_selection(self) -> None:
        score = np.array([[0.9, 0.8], [0.2, 0.1]], dtype=np.float32)
        gt = np.array([[1, 1], [0, 0]], dtype=bool)
        accumulator = PixelMetricAccumulator(threshold=0.5, bins=32)
        accumulator.update(score, gt, np.ones_like(gt))
        metrics = accumulator.compute()
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertAlmostEqual(metrics["ap"], 1.0)
        overridden = PixelMetricAccumulator(threshold=0.5, bins=32)
        overridden.update(score, gt, np.ones_like(gt), binary_pred=np.zeros_like(gt))
        overridden_metrics = overridden.compute()
        self.assertAlmostEqual(overridden_metrics["ap"], 1.0)
        self.assertAlmostEqual(overridden_metrics["f1"], 0.0)
        samples = [
            Sample(f"{source}__{index}", __file__, __file__)
            for source in ("road_anomaly", "smiyc", "street_hazards")
            for index in range(3)
        ]
        selected = select_samples(samples, 3, "stratified")
        self.assertEqual({item.sample_id.split("__")[0] for item in selected}, {"road_anomaly", "smiyc", "street_hazards"})
        selected_ten = select_samples(samples, 8, "stratified")
        counts = {source: 0 for source in ("road_anomaly", "smiyc", "street_hazards")}
        for item in selected_ten:
            counts[item.sample_id.split("__")[0]] += 1
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        calibration = select_samples(samples, 3, "stratified", offset=0)
        validation = select_samples(samples, 3, "stratified", offset=3)
        self.assertTrue({item.sample_id for item in calibration}.isdisjoint(item.sample_id for item in validation))


if __name__ == "__main__":
    unittest.main()
