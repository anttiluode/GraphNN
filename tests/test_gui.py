import os
import tempfile
import time
import unittest
from pathlib import Path

import tkinter as tk
import numpy as np
import torch

from graphnn.gui import JanusApp
from graphnn.model import ModelConfig
from graphnn.training import PairSession


@unittest.skipUnless(os.environ.get("DISPLAY") or os.name == "nt", "A display or Xvfb is required")
class GuiTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.root = tk.Tk()
        self.session = PairSession(ModelConfig(image_size=16,grid_size=4,channels=4,width=12),device="cpu")
        self.app = JanusApp(self.root,self.session)
        self.pump(.15)

    def tearDown(self):
        if not self.app.closed:
            self.app.close()

    def pump(self,seconds):
        until = time.monotonic()+seconds
        while time.monotonic()<until and not self.app.closed:
            self.root.update()
            time.sleep(.005)

    def wait_idle(self):
        until = time.monotonic()+10
        while self.app.busy and time.monotonic()<until:
            self.pump(.05)
        self.assertFalse(self.app.busy)

    def test_slider_and_real_training_controls(self):
        self.app.angle.set(90.)
        self.app.request_render()
        self.pump(.2)
        self.assertIn("90",self.app.query_text.get())
        self.app.step_limit.set("3")
        self.app.train_button.invoke()
        self.wait_idle()
        self.assertEqual(self.session.steps,3)
        self.assertTrue(self.app.display_images)

    def test_pause_and_close_during_training(self):
        self.app.step_limit.set("10000")
        self.app.train_button.invoke()
        self.pump(.1)
        self.app.train_button.invoke()
        self.wait_idle()
        self.assertLess(self.session.steps,10000)
        self.app.train_button.invoke()
        self.app.close()
        self.assertTrue(self.app.closed)

    def test_checkpoint_load_and_control_preserve_functioning_window(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"memory.pt"
            self.session.save(path)
            self.app.load_checkpoint(path)
            self.wait_idle()
            self.pump(.1)
            self.assertIsNone(self.session.images)
            self.app.control.set("randomized")
            self.app.request_render()
            self.pump(.15)
            self.assertTrue(self.app.display_images)

    def test_explicit_writes_and_patch_run_with_frozen_weights(self):
        self.app.write_buttons[0].invoke()
        self.wait_idle()
        before={n:p.clone() for n,p in self.session.models["graph"].state_dict().items()}
        self.app.rewrite_button.invoke()
        self.wait_idle()
        self.pump(.1)
        self.assertIn("b_mse_after",self.session.last_write_report["graph"])
        for n,p in self.session.models["graph"].state_dict().items(): self.assertTrue(torch.equal(p,before[n]))

    def test_loading_sources_stages_without_replaying_b(self):
        from graphnn.data import demo_pair,image_from_array
        prior=self.session.memories["graph"].flat().clone()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"a.png"
            image_from_array(demo_pair(16,127)[0]).save(p)
            self.app.load_source(0,p)
            self.wait_idle()
            self.assertTrue(torch.equal(prior,self.session.memories["graph"].flat()))
            self.app.write_buttons[0].invoke()
            self.wait_idle()
            self.assertFalse(torch.equal(prior,self.session.memories["graph"].flat()))

    def test_rewrite_fit_keeps_following_pair_fit_on_displayed_sources(self):
        original=self.session.images.copy()
        self.app.fit_task.set("rewrite")
        self.app.step_limit.set("2")
        self.app.train_button.invoke()
        self.wait_idle()
        shown=np.stack(self.app.image_slots)
        self.assertEqual(self.session.mode,"rewrite-fit")
        self.assertFalse(np.array_equal(shown[0],original[0]))
        self.assertTrue(np.array_equal(shown[1],original[1]))
        self.assertTrue(np.array_equal(self.session.images,shown))
        self.app.fit_task.set("pair")
        self.app.step_limit.set("1")
        self.app.train_button.invoke()
        self.wait_idle()
        self.assertTrue(np.array_equal(np.stack(self.session.memory_targets),shown))


if __name__ == "__main__":
    unittest.main()
