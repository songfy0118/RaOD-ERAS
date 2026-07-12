from __future__ import annotations

import unittest

import numpy as np

from src.raod_eras.datasets import Sample
from src.raod_eras.experiment import select_samples
from src.raod_eras.metrics import PixelMetricAccumulator, evaluate_binary
from src.raod_eras.object_refinement import AnomalyInstance, refine_objects
from src.raod_eras.risk_planning import plan_risk_response


class ClosedLoopTests(unittest.TestCase):
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

    def test_streaming_metrics_and_stratified_selection(self) -> None:
        score = np.array([[0.9, 0.8], [0.2, 0.1]], dtype=np.float32)
        gt = np.array([[1, 1], [0, 0]], dtype=bool)
        accumulator = PixelMetricAccumulator(threshold=0.5, bins=32)
        accumulator.update(score, gt, np.ones_like(gt))
        metrics = accumulator.compute()
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertAlmostEqual(metrics["ap"], 1.0)
        samples = [
            Sample(f"{source}__{index}", __file__, __file__)
            for source in ("road_anomaly", "smiyc", "street_hazards")
            for index in range(3)
        ]
        selected = select_samples(samples, 3, "stratified")
        self.assertEqual({item.sample_id.split("__")[0] for item in selected}, {"road_anomaly", "smiyc", "street_hazards"})


if __name__ == "__main__":
    unittest.main()
