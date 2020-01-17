import unittest
import sys
sys.path.append('.')
from data.datasets.naic import NaicDataset, NaicTest


class DatasetTestCase(unittest.TestCase):
    def test_naic_dataset(self):
        d1 = NaicDataset()
        d2 = NaicDataset()
        for i in range(len(d1.query)):
            assert d1.query[i][0] == d2.query[i][0]

    def test_naic_testdata(self):
        test_dataset = NaicTest()


if __name__ == '__main__':
    unittest.main()
