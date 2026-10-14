import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from graphnn.data import demo_pair, procedural_pairs
from graphnn.model import ModelConfig
from graphnn.training import PairSession, evaluate


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.session = PairSession(ModelConfig(image_size=16, grid_size=4, channels=4, width=12), device="cpu", seed=7)
        self.session.set_pair(demo_pair(16, 7))

    def test_fitting_reduces_both_losses(self):
        initial = self.session.metrics()
        for _ in range(35):
            self.session.step()
        final = self.session.metrics()
        for arm in ("graph", "vector"):
            self.assertLess(final[arm]["mse"], initial[arm]["mse"] * .9)

    def test_saved_memory_renders_without_original_images(self):
        self.session.step()
        expected = self.session.render(90.0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "memory.pt"
            self.session.save(path)
            payload = torch.load(path, weights_only=True)
            self.assertNotIn("images", payload)
            self.assertNotIn("targets", payload)
            restored = PairSession(device="cpu")
            restored.load(path)
            self.assertIsNone(restored.images)
            actual = restored.render(90.0)
            for arm in expected:
                np.testing.assert_array_equal(actual[arm], expected[arm])
            with self.assertRaises(ValueError):
                restored.step()

    def test_frozen_encoding_does_not_change_network_weights(self):
        before = {k: v.clone() for k,v in self.session.models["graph"].state_dict().items()}
        self.session.set_pair(demo_pair(16, 97))
        self.session.encode()
        for k,v in self.session.models["graph"].state_dict().items():
            self.assertTrue(torch.equal(v,before[k]))

    def test_memory_controls_change_answer_and_leave_stored_memory_intact(self):
        before = self.session.memories["graph"].flat().clone()
        original = self.session.render(0.)["graph"]
        random = self.session.render(0., "randomized")["graph"]
        self.assertGreater(np.abs(original-random).max(), 1e-5)
        self.assertTrue(torch.equal(self.session.memories["graph"].flat(), before))
        np.testing.assert_array_equal(random, self.session.render(0., "randomized")["graph"])

    def test_bad_checkpoint_does_not_destroy_session(self):
        expected = self.session.render(0.)["graph"]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"bad.pt"
            torch.save({"format": "graphnn-memory", "version": 1, "config": {"image_size": 100000}}, path)
            with self.assertRaises(ValueError):
                self.session.load(path)
        np.testing.assert_array_equal(self.session.render(0.)["graph"], expected)

    def test_checkpoint_cast_overflow_is_rejected_atomically(self):
        expected = self.session.render(0.)["graph"]
        images = self.session.images.copy()
        original_model = self.session.models["graph"]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"overflow.pt"
            self.session.save(path)
            payload = torch.load(path,weights_only=True)
            tensor = payload["models"]["graph"]["decoder.4.weight"].double()
            tensor.fill_(1e300)
            payload["models"]["graph"]["decoder.4.weight"] = tensor
            torch.save(payload,path)
            with self.assertRaises(ValueError):
                self.session.load(path)
        self.assertIs(self.session.models["graph"],original_model)
        np.testing.assert_array_equal(self.session.images,images)
        np.testing.assert_array_equal(self.session.render(0.)["graph"],expected)

    def test_data_and_evaluation_are_reproducible_and_charge_bytes(self):
        pairs = procedural_pairs(4,16,97)
        self.assertTrue(torch.equal(pairs,procedural_pairs(4,16,97)))
        results = evaluate(self.session.models,pairs)
        self.assertEqual(results["graph"]["payload_bytes"],192)
        self.assertIn("shuffled_mse",results["graph"])
        self.assertTrue(np.isfinite(results["vector"]["mse"]))


if __name__ == "__main__":
    unittest.main()
