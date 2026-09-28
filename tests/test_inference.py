import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from src.raod_eras.inference import anomaly_score, predict_image
from src.raod_eras.score_to_mask import PromptResult


class InferenceTests(unittest.TestCase):
    def test_uniform_scene_has_finite_image_sized_score(self):
        class Encoder:
            input_size = 56

            def patch_features(self, image, input_size=None):
                return np.ones((4, 4, 8), dtype=np.float32)

        score = anomaly_score(Encoder(), Image.new("RGB", (60, 40), (128, 128, 128)))
        self.assertEqual(score.shape, (40, 60))
        self.assertTrue(np.isfinite(score).all())
        self.assertGreaterEqual(float(score.min()), 0)
        self.assertLessEqual(float(score.max()), 1)

    def test_invalid_score_is_rejected_before_sam(self):
        predictor = Mock()
        with patch("src.raod_eras.inference.anomaly_score", return_value=np.full((40, 60), np.nan)):
            with self.assertRaisesRegex(ValueError, "finite"):
                predict_image(object(), predictor, Image.new("RGB", (60, 40)))
        predictor.set_image.assert_not_called()

    def test_inference_passes_image_and_score_without_ground_truth(self):
        image = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        score = np.full((40, 60), 0.25, dtype=np.float32)
        result = PromptResult(np.zeros((40, 60), dtype=bool), score)
        class Predictor:
            def set_image(self, rgb):
                self.image = rgb
        predictor = Predictor()
        with patch("src.raod_eras.inference.anomaly_score", return_value=score), patch(
            "src.raod_eras.inference.sam_box_ablation", return_value=result
        ) as method:
            actual_score, actual = predict_image(object(), predictor, image)
        self.assertIs(actual, result)
        np.testing.assert_array_equal(actual_score, score)
        self.assertEqual(predictor.image.shape, (40, 60, 3))
        self.assertEqual(method.call_args.kwargs, dict(road_aware=True, boundary_aware=True, feedback=True))

    def test_help_and_missing_inputs_do_not_load_weights(self):
        script = Path(__file__).resolve().parents[1] / "scripts/predict_image.py"
        help_result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        result = subprocess.run([sys.executable, str(script), "--image", "not-present.png",
                                 "--sam-checkpoint", "not-present.pth"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Image not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
