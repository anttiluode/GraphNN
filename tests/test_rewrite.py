import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from graphnn.data import demo_pair, procedural_pairs
from graphnn.model import ModelConfig
from graphnn.training import PairSession
from graphnn.rewrite import make_rewrite_batch, evaluate_rewrites, fit_rewrite_step


class RewriteTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg=ModelConfig(image_size=16,grid_size=4,channels=4,width=12)
        self.session=PairSession(self.cfg,device="cpu",seed=41)
        self.session.set_pair(demo_pair(16,41))

    def test_examples_change_only_a_mark_and_keep_b_identical(self):
        batch=make_rewrite_batch(procedural_pairs(4,16,97),seed=41)
        self.assertTrue(torch.equal(batch.updated[:,1],batch.original[:,1]))
        outside=(1-batch.masks[:,0]).expand(-1,3,-1,-1)
        self.assertTrue(torch.equal(batch.updated[:,0]*outside,batch.original[:,0]*outside))
        self.assertTrue(torch.equal(batch.new_a_labels,1-batch.labels[:,0]))
        self.assertGreater((batch.updated[:,0]-batch.original[:,0]).abs().sum().item(),0)

    def test_single_write_uses_previous_memory_without_reencoding_b(self):
        before={k:{n:t.clone() for n,t in m.state_dict().items()} for k,m in self.session.models.items()}
        calls=[]
        original=self.session.models["graph"].write
        def observe(image,context,previous=None):
            calls.append((context,previous))
            return original(image,context,previous)
        self.session.models["graph"].write=observe
        prior=self.session.memories["graph"]
        new_a=demo_pair(16,127)[0]
        self.session.write_image(new_a,0.)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0],0.)
        self.assertIs(calls[0][1],prior)
        for arm,model in self.session.models.items():
            for n,t in model.state_dict().items(): self.assertTrue(torch.equal(t,before[arm][n]))
        self.assertEqual(self.session.memories["graph"].flat().dtype,torch.uint8)

    def test_staging_does_not_change_memory_or_written_targets(self):
        memory=self.session.memories["graph"].flat().clone()
        target=self.session.memory_targets[0].copy()
        self.session.set_pair(demo_pair(16,127),encode=False)
        self.assertTrue(torch.equal(memory,self.session.memories["graph"].flat()))
        np.testing.assert_array_equal(target,self.session.memory_targets[0])

    def test_source_free_checkpoint_allows_a_write_without_b_pixels(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"memory.pt";self.session.save(p)
            restored=PairSession(device="cpu");restored.load(p)
            restored.write_image(demo_pair(16,97)[0],0.)
            self.assertIsNone(restored.images)
            self.assertIsNone(restored.memory_targets[1])
            self.assertIn("A",restored.metrics()["graph"]["endpoint_mse"])
            self.assertNotIn("B",restored.metrics()["graph"]["endpoint_mse"])
            self.assertEqual(restored.render(90.)["graph"].shape,(16,16,3))

    def test_gate_metrics_include_retention_and_controls(self):
        batch=make_rewrite_batch(procedural_pairs(4,16,97),seed=127)
        result=evaluate_rewrites(self.session.models,batch)
        self.assertEqual(result["graph"]["payload_bytes"],192)
        for condition in ("before","after","no_write","shuffled","rebuilt"):
            self.assertIn(condition,result["graph"])
        self.assertIn("joint_mark_accuracy",result["graph"]["after"])
        self.assertIn("b_prediction_change_mse",result["graph"])

    def test_frozen_pair_rebuilds_quantize_before_the_next_write(self):
        batch=make_rewrite_batch(procedural_pairs(4,16,97),seed=127)
        observed=[]
        model=self.session.models["graph"]
        write=model.write
        def observe(image,context,previous=None):
            if previous is not None:
                observed.append(previous.flat().dtype)
            return write(image,context,previous)
        model.write=observe
        for name,operation in (("evaluation",lambda:evaluate_rewrites(self.session.models,batch)),
                               ("GUI pair rebuild",self.session.encode)):
            with self.subTest(operation=name):
                observed.clear()
                operation()
                self.assertTrue(observed)
                self.assertTrue(all(dtype==torch.uint8 for dtype in observed),observed)

    def test_rewrite_loss_learns_and_session_patch_reports_retention(self):
        batch=make_rewrite_batch(procedural_pairs(2,16,97),seed=127)
        first=fit_rewrite_step(self.session.models,self.session.optimizers,batch)
        for _ in range(30): last=fit_rewrite_step(self.session.models,self.session.optimizers,batch)
        self.assertLess(last["graph"],first["graph"])
        self.assertLess(last["vector"],first["vector"])
        report=self.session.rewrite_a()
        self.assertIn("b_mse_before",report["graph"])
        self.assertIn("b_mse_after",report["graph"])
        self.assertGreater(report["graph"]["changed_bytes"],0)


if __name__=="__main__": unittest.main()
