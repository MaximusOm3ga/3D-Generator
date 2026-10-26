import unittest
import torch
from implicit3d.model import SDFAutoencoder

class ImplicitTests(unittest.TestCase):
    def test_sdf_shape_and_gradients(self):
        model=SDFAutoencoder(num_classes=4,latent_dim=16,width=64,encoder_width=32,fourier_bands=3)
        points=torch.randn(2,64,3); query=torch.randn(2,128,3); labels=torch.tensor([0,2])
        pred,z=model(points,query,labels)
        self.assertEqual(tuple(pred.shape),(2,128))
        self.assertEqual(tuple(z.shape),(2,16))
        loss=pred.square().mean(); loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
    def test_decoder_query_chunks(self):
        model=SDFAutoencoder(num_classes=2,latent_dim=8,width=32,encoder_width=16,fourier_bands=2)
        pred=model.decoder(torch.zeros(1,10,3),torch.zeros(1,8),torch.tensor([1]))
        self.assertEqual(tuple(pred.shape),(1,10))

if __name__=="__main__": unittest.main()
