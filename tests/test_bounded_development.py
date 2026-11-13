"""Development-only intervention checks; not evidence of diagnostic validity."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from ml.bounded_development import ablate


class BoundedDevelopmentTests(unittest.TestCase):
    def test_interventions_target_opposite_regions_without_changing_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'image.png'
            values=np.full((100,100),100,dtype=np.uint8)
            values[:10]=0;values[-10:]=0;values[:,:10]=0;values[:,-10:]=0
            Image.fromarray(values).save(path)
            border=ablate(path,'border_removed').reshape(32,32)
            center=ablate(path,'center_removed').reshape(32,32)
            self.assertTrue(np.isfinite(border).all())
            self.assertAlmostEqual(float(border.min()),100/255)
            self.assertLess(float(center[0,0]),float(border[0,0]))


if __name__=='__main__':unittest.main()
