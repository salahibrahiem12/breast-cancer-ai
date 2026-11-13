"""Non-mutating checks of the reserved-cohort inference guards."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ml import busbra_locked


class LockedEvaluationTests(unittest.TestCase):
    def test_second_inference_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'INFERENCE_STARTED.json').write_text('{}')
            with patch.object(busbra_locked,'ROOT',root),patch.object(busbra_locked,'encoder') as model:
                with self.assertRaises(FileExistsError):busbra_locked.evaluate()
                model.assert_not_called()

    def test_modified_lock_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'lock.json').write_text('{}')
            (root/'LOCK_SHA256.txt').write_text('incorrect')
            with patch.object(busbra_locked,'ROOT',root),patch.object(busbra_locked,'encoder') as model:
                with self.assertRaisesRegex(ValueError,'Lock modified'):busbra_locked.evaluate()
                model.assert_not_called()
                self.assertFalse((root/'INFERENCE_STARTED.json').exists())


if __name__=='__main__':unittest.main()
