"""Numerical checks for exploratory source-held-out evaluation."""
import unittest
import numpy as np
from ml.cross_source import aggregate,metric,threshold


class CrossSourceTests(unittest.TestCase):
    def test_operating_threshold_changes_counts_not_auc_or_brier(self):
        result=metric(np.array([0,0,1,1]),np.array([.1,.6,.4,.9]),.8)
        self.assertEqual([result[k] for k in ['tn','fp','fn','tp']],[2,0,1,1])
        self.assertAlmostEqual(result['roc_auc'],.75)
        self.assertAlmostEqual(result['binary_brier'],.185)
        self.assertAlmostEqual(result['balanced_accuracy'],.75)

    def test_group_average_counts_each_recorded_group_once(self):
        y,p,g=aggregate(np.array([0,0,1]),np.array([.1,.3,.8]),np.array(['a','a','b']))
        self.assertEqual(y.tolist(),[0,1]);np.testing.assert_allclose(p,[.2,.8])
        self.assertEqual(g.tolist(),['a','b'])

    def test_validation_threshold_is_deterministic_and_separates_known_groups(self):
        y=np.array([0,0,1,1]);p=np.array([.1,.2,.8,.9]);g=np.array(['a','a','b','b'])
        first=threshold(y,p,g);self.assertEqual(first,threshold(y,p,g))
        self.assertAlmostEqual(metric(*aggregate(y,p,g)[:2],first)['balanced_accuracy'],1)


if __name__=='__main__':unittest.main()
