import unittest
import torch
from conditional3d.hierarchical import (
    HierarchicalPointDenoiser, DiffusionSchedule, diffusion_loss
)


class HierarchicalTests(unittest.TestCase):
    def test_shape_and_loss(self):
        model=HierarchicalPointDenoiser(num_classes=4,points=64,width=32,depth=2)
        x=torch.randn(2,64,3); t=torch.randint(0,20,(2,)); y=torch.tensor([0,2])
        self.assertEqual(tuple(model(x,t,y).shape),(2,64,3))
        schedule=DiffusionSchedule(20)
        loss=diffusion_loss(model,schedule,x,y)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()

    def test_null_class(self):
        model=HierarchicalPointDenoiser(num_classes=3,points=32,width=32,depth=1)
        out=model(torch.randn(1,32,3),torch.tensor([3]),torch.tensor([model.null_class]))
        self.assertEqual(tuple(out.shape),(1,32,3))

if __name__=="__main__": unittest.main()
