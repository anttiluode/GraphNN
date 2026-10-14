import os
import tempfile
import time
import unittest
from pathlib import Path

import tkinter as tk
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


if __name__ == "__main__":
    unittest.main()
