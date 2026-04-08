"""Unit tests for CoverageEstimator — pure numpy, no file I/O."""
import numpy as np
import pytest
from scipy import stats as sps

from cleansweep.coverage import CoverageEstimator


class TestFitNbinomMLE:

    def test_returns_positive_r(self):
        depths = np.array([40, 50, 60, 55, 45, 58, 62, 70], dtype=float)
        r, p = CoverageEstimator()._fit_nbinom_mle(depths)
        assert r > 0.0

    def test_returns_valid_p(self):
        depths = np.array([40, 50, 60, 55, 45, 58, 62, 70], dtype=float)
        _, p = CoverageEstimator()._fit_nbinom_mle(depths)
        assert 0.0 < p < 1.0

    @pytest.mark.parametrize("true_r,true_p,n", [
        (5.0,  0.40, 3000),
        (10.0, 0.15, 3000),
        (2.0,  0.30, 5000),
    ])
    def test_recovers_known_parameters(self, true_r, true_p, n):
        rng = np.random.default_rng(0)
        depths = rng.negative_binomial(true_r, true_p, size=n).astype(float)
        r, p = CoverageEstimator()._fit_nbinom_mle(depths)
        assert abs(r - true_r) / true_r < 0.20
        assert abs(p - true_p) / true_p < 0.20

    def test_fitted_mean_matches_sample_mean(self):
        rng = np.random.default_rng(7)
        depths = rng.negative_binomial(8, 0.12, size=4000).astype(float)
        r, p = CoverageEstimator()._fit_nbinom_mle(depths)
        fitted_mean = sps.nbinom(r, p).mean()
        sample_mean = float(np.mean(depths))
        assert abs(fitted_mean - sample_mean) / sample_mean < 0.05

    def test_low_variance_data_does_not_raise(self):
        """Near-constant depths (var ≈ 0) must not crash the optimizer."""
        rng = np.random.default_rng(0)
        depths = np.full(200, 50.0) + rng.normal(0, 0.1, 200)
        r, p = CoverageEstimator()._fit_nbinom_mle(depths)
        assert r > 0 and 0 < p < 1

    def test_small_sample_does_not_raise(self):
        depths = np.array([30.0, 40.0, 50.0, 60.0, 55.0])
        r, p = CoverageEstimator()._fit_nbinom_mle(depths)
        assert r > 0 and 0 < p < 1

    def test_underdispersed_data_does_not_raise(self):
        """var < mean causes MoM to give r < 0; MLE must still return valid params."""
        # All-same values → variance = 0
        depths = np.full(500, 50.0)
        r, p = CoverageEstimator()._fit_nbinom_mle(depths)
        assert r > 0 and 0 < p < 1


class TestCoverageEstimatorEstimate:

    def test_returns_median_as_coverage(self):
        depths = np.array([10, 20, 30, 40, 50], dtype=float)
        coverage, _ = CoverageEstimator().estimate(depths)
        assert coverage == 30.0

    def test_no_distribution_when_use_mle_false(self):
        depths = np.array([40, 50, 60, 55, 45], dtype=float)
        _, dist = CoverageEstimator().estimate(depths, use_mle=False)
        assert dist is None

    def test_returns_nbinom_dist_when_use_mle_true(self):
        rng = np.random.default_rng(0)
        depths = rng.negative_binomial(5, 0.1, size=500).astype(float)
        _, dist = CoverageEstimator().estimate(depths, use_mle=True)
        assert dist is not None
        assert hasattr(dist, "pmf") and hasattr(dist, "cdf")

    def test_sets_r_and_p_when_use_mle_true(self):
        rng = np.random.default_rng(1)
        depths = rng.negative_binomial(5, 0.1, size=500).astype(float)
        ce = CoverageEstimator()
        ce.estimate(depths, use_mle=True)
        assert hasattr(ce, "r") and hasattr(ce, "p")
        assert ce.r > 0 and 0 < ce.p < 1

    def test_does_not_set_r_p_when_use_mle_false(self):
        depths = np.array([40, 50, 60], dtype=float)
        ce = CoverageEstimator()
        ce.estimate(depths, use_mle=False)
        assert not hasattr(ce, "r")
        assert not hasattr(ce, "p")
