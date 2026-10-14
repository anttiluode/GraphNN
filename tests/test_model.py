import unittest

import torch

from graphnn.model import GraphMemory, JanusModel, ModelConfig


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.cfg = ModelConfig(image_size=16, grid_size=4, channels=4, width=12)
        self.model = JanusModel(self.cfg)
        self.pair = torch.rand(2, 2, 3, 16, 16)

    def test_memory_bytes_and_vector_control_match(self):
        memory = self.model.encode_pair(self.pair).quantized()
        self.assertEqual(memory.flat().dtype, torch.uint8)
        self.assertEqual(memory.flat().shape, (2, 192))
        self.assertEqual(self.model.payload_bytes, 192)
        self.assertEqual(memory.flat().numel() * memory.flat().element_size(), 384)
        vector = JanusModel(self.cfg, reader="vector")
        self.assertEqual(vector.payload_bytes, self.model.payload_bytes)

    def test_fresh_reads_and_query_dependence(self):
        memory = self.model.encode_pair(self.pair).quantized()
        before = memory.flat().clone()
        a = self.model.read(memory, 0.0)
        b = self.model.read(memory, 90.0)
        self.assertEqual(a.shape, (2, 3, 16, 16))
        self.assertTrue(torch.equal(a, self.model.read(memory, 0.0)))
        self.assertTrue(torch.equal(before, memory.flat()))
        self.assertGreater((a - b).abs().max().item(), 1e-5)
        self.assertTrue(bool(((a >= 0) & (a <= 1)).all()))

    def test_writer_and_transport_receive_gradients(self):
        memory = self.model.encode_pair(self.pair)
        loss = (self.model.read(memory, 0.0) - self.pair[:, 0]).square().mean()
        loss.backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in self.model.writer.parameters() if p.grad is not None), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_previous_memory_is_not_modified_in_place(self):
        first = self.model.write(self.pair[:, 0], 0.0).quantized()
        original = first.flat().clone()
        second = self.model.write(self.pair[:, 1], 90.0, previous=first).quantized()
        self.assertTrue(torch.equal(first.flat(), original))
        self.assertFalse(torch.equal(first.flat(), second.flat()))

    def test_bad_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            ModelConfig(image_size=0)
        with self.assertRaises(ValueError):
            self.model.encode_pair(self.pair[:, :1])
        with self.assertRaises(ValueError):
            self.model.write(torch.full((1, 3, 16, 16), float("nan")), 0.0)
        memory = self.model.encode_pair(self.pair).quantized()
        for angle in (-1, 91, float("nan")):
            with self.assertRaises(ValueError):
                self.model.read(memory, angle)
        with self.assertRaises(ValueError):
            self.model.read(GraphMemory(tuple(t[..., :1] for t in memory.parts)), 0.0)


if __name__ == "__main__":
    unittest.main()
