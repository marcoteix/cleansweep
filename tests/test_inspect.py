"""Unit tests for Inspector.cleansweep_info() — mock VCFFilter objects, no I/O."""
import pytest
from scipy import stats as sps

from cleansweep.coverage import CoverageEstimator
from cleansweep.filter import VCFFilter
from cleansweep.inspect import Inspector
from cleansweep.mcmc import BaseCountFilter


@pytest.fixture
def fast_filter():
    """VCFFilter fitted with method='fast' — NB params from MLE."""
    vf = VCFFilter(random_state=42)
    vf.method = "fast"
    vf.query_coverage = 57.5

    ce = CoverageEstimator(random_state=42)
    ce.r = 5.0
    ce.p = 0.08
    vf.coverage_estimator = ce

    return vf


@pytest.fixture
def mixture_filter():
    """VCFFilter fitted with method='mixture' — params from mocked MCMC."""
    vf = VCFFilter(random_state=42)
    vf.method = "mixture"
    vf.query_coverage = 57.5

    bf = BaseCountFilter(chains=1, draws=10, burn_in=1, threads=1)
    bf.dist_params = {
        "query_overdispersion": 200.0,
        "alt_prob": 0.35,
    }
    vf.basecount_filter = bf

    return vf


class TestCleanSweepInfoFast:

    def test_returns_dict(self, fast_filter):
        assert isinstance(Inspector().cleansweep_info(fast_filter), dict)

    def test_method_reported(self, fast_filter):
        assert Inspector().cleansweep_info(fast_filter)["Fitting method"] == "fast"

    def test_coverage_reported(self, fast_filter):
        result = Inspector().cleansweep_info(fast_filter)
        assert result["Estimated mean query depth of coverage"] == pytest.approx(57.5)

    def test_random_seed_reported(self, fast_filter):
        assert Inspector().cleansweep_info(fast_filter)["Random seed"] == 42

    def test_nb_parameters_present(self, fast_filter):
        nb = Inspector().cleansweep_info(fast_filter)["Negative Binomial parameters"]
        for key in ("r", "p", "mean", "std", "percentile_2.5", "percentile_97.5"):
            assert key in nb

    def test_nb_r_and_p_correct(self, fast_filter):
        nb = Inspector().cleansweep_info(fast_filter)["Negative Binomial parameters"]
        assert nb["r"] == pytest.approx(5.0)
        assert nb["p"] == pytest.approx(0.08)

    def test_nb_mean_consistent_with_scipy(self, fast_filter):
        nb = Inspector().cleansweep_info(fast_filter)["Negative Binomial parameters"]
        expected = sps.nbinom(nb["r"], nb["p"]).mean()
        assert nb["mean"] == pytest.approx(expected)

    def test_nb_percentiles_ordered(self, fast_filter):
        nb = Inspector().cleansweep_info(fast_filter)["Negative Binomial parameters"]
        assert nb["percentile_2.5"] <= nb["percentile_97.5"]

    def test_no_mcmc_options_for_fast(self, fast_filter):
        result = Inspector().cleansweep_info(fast_filter)
        assert "MCMC options" not in result


class TestCleanSweepInfoMixture:

    def test_mcmc_options_present(self, mixture_filter):
        result = Inspector().cleansweep_info(mixture_filter)
        assert "MCMC options" in result

    def test_overdispersion_key_present(self, mixture_filter):
        result = Inspector().cleansweep_info(mixture_filter)
        keys_with_overdispersion = [k for k in result if "overdispersion" in k.lower()]
        assert len(keys_with_overdispersion) == 1

    def test_nb_r_derived_from_alpha(self, mixture_filter):
        """r = alpha = query_overdispersion."""
        nb = Inspector().cleansweep_info(mixture_filter)["Negative Binomial parameters"]
        assert nb["r"] == pytest.approx(200.0)

    def test_nb_p_derived_correctly(self, mixture_filter):
        """p = alpha / (alpha + mu)."""
        alpha, mu = 200.0, 57.5
        expected_p = alpha / (alpha + mu)
        nb = Inspector().cleansweep_info(mixture_filter)["Negative Binomial parameters"]
        assert nb["p"] == pytest.approx(expected_p)

    def test_nb_mean_consistent_with_scipy(self, mixture_filter):
        nb = Inspector().cleansweep_info(mixture_filter)["Negative Binomial parameters"]
        expected = sps.nbinom(nb["r"], nb["p"]).mean()
        assert nb["mean"] == pytest.approx(expected)
