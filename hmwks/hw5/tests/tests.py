import unittest
import numpy as np


class TestPCA(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array([[72,  4, 24, 70],[41, 43, 80, 78],[62, 19, 64, 85], [15, 45, 41, 33],[35,  6, 31, 82]])
        self.pca = PCA(0.99)
        
    def test_mean_shape(self):
        self.assertEqual(self.pca.compute_mean_vector(self.X_train).shape,(4,))

    def test_cov_shape(self):
        self.assertEqual(self.pca.compute_cov(self.X_train,self.pca.compute_mean_vector(self.X_train)).shape,(4,4))

    def test_reduced_shape(self):
        ret = self.pca.fit(self.X_train)
        self.assertEqual(ret.shape, (5,2))

    def test_explained_variance(self):
        eigen_vals = [2.84574523, 1.72946803, 0.41785852, 0.00692822]
        actual = [0.5691490465397107, 0.34589360551199344, 0.083571703784301, 0.0013856441639950483]
        returned = self.pca.compute_explained_variance(eigen_vals)
        np.testing.assert_almost_equal(returned, actual, 5)

if __name__ == '__main__':
    unittest.main()