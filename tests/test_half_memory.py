"""Constrained-storage and graph transport tests for phase computer Gate 2."""
import unittest
import torch
from graphnn.model import ModelConfig
from graphnn.half_memory import HalfJanus, HalfMemory
from graphnn.rewrite import make_rewrite_batch, fit_rewrite_step
from graphnn.data import procedural_pairs


class HalfPlaneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg=ModelConfig(image_size=16,grid_size=4,channels=4,width=12)
        torch.manual_seed(5)

    def test_physical_bytes_are_half_and_uint8(self):
        for kind,reader in (("phase_only","graph"),("gain_only","graph"),("phase_only","vector")):
            with self.subTest(kind=kind,reader=reader):
                m=HalfJanus(self.cfg,kind=kind,reader=reader)
                pair=torch.rand(2,2,3,16,16)
                state=m.encode_pair(pair).quantized()
                self.assertEqual(m.payload_bytes,96)
                self.assertEqual(state.flat().shape,(2,96))
                self.assertEqual(state.flat().dtype,torch.uint8)
                self.assertEqual(len(state.parts),2)
                self.assertEqual(m.read(state,0.).shape,(2,3,16,16))

    def test_hidden_constant_is_not_saved_as_state(self):
        for kind in ("gain_only","phase_only"):
            m=HalfJanus(self.cfg,kind=kind)
            p=m.write(torch.rand(1,3,16,16),0.).quantized()
            h,ph,v,pv=m.expanded(p).parts
            if kind=="phase_only":
                self.assertTrue(torch.all(h==0.5))
                self.assertTrue(torch.all(v==0.5))
                self.assertTrue(torch.equal(ph,p.parts[0].float()/255))
                self.assertTrue(torch.equal(pv,p.parts[1].float()/255))
            else:
                self.assertTrue(torch.all(ph==0.5))
                self.assertTrue(torch.all(pv==0.5))
                self.assertTrue(torch.equal(h,p.parts[0].float()/255))
                self.assertTrue(torch.equal(v,p.parts[1].float()/255))

    def test_phase_and_gain_share_the_same_parameter_count(self):
        a=HalfJanus(self.cfg,kind="gain_only")
        b=HalfJanus(self.cfg,kind="phase_only")
        c=HalfJanus(self.cfg,kind="phase_only",reader="vector")
        counts=[sum(t.numel() for t in m.parameters()) for m in (a,b,c)]
        self.assertEqual(len(set(counts)),1)
        self.assertLess(counts[0],sum(t.numel() for t in __import__("graphnn.model",fromlist=["JanusModel"]).JanusModel(self.cfg).parameters()))

    def test_real_training_updates_half_state_and_has_finite_gradients(self):
        models={"phase_only_896":HalfJanus(self.cfg),"gain_only_896":HalfJanus(self.cfg,kind="gain_only")}
        opt={k:torch.optim.Adam(m.parameters(),lr=.003) for k,m in models.items()}
        batch=make_rewrite_batch(procedural_pairs(4,size=16,seed=12),seed=17)
        result=fit_rewrite_step(models,opt,batch)
        for k,m in models.items():
            self.assertTrue(0<result[k]<1)
            self.assertTrue(all(torch.isfinite(p).all() for p in m.parameters()))
            a=m.write(batch.original[:,0],0.).quantized()
            ab=m.write(batch.original[:,1],90.,a).quantized()
            updated=m.write(batch.updated[:,0],0.,ab).quantized()
            self.assertEqual(updated.flat().dtype,torch.uint8)
            self.assertFalse(torch.equal(ab.flat(),updated.flat()))

    def test_no_extra_per_example_reader_activations_after_save(self):
        m=HalfJanus(self.cfg)
        img=torch.rand(2,3,16,16)
        state=m.write(img,0.).quantized()
        before=state.flat().clone()
        self.assertTrue(torch.equal(m.read(state,90.),m.read(state,90.)))
        self.assertTrue(torch.equal(before,state.flat()))


if __name__=="__main__":
    unittest.main()
