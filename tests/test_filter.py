"""
Tests for the filter pipeline.

Tier 1 — unit tests (pure Python, no subprocess, no file I/O).
Tier 2 — integration tests using synthetic fixtures from conftest.py.
"""
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from cleansweep.coverage import CoverageEstimator
from cleansweep.filter import NucmerSNPFilter
from cleansweep.mcmc import BaseCountFilter


# ---------------------------------------------------------------------------
# Tier 1 — Unit tests
# ---------------------------------------------------------------------------

class TestCoverageEstimatorEstimate:
    """Focused subset; full coverage is in test_coverage.py."""

    def test_median_returned_as_coverage(self):
        depths = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        coverage, _ = CoverageEstimator().estimate(depths)
        assert coverage == 30.0

    def test_no_dist_when_use_mle_false(self):
        depths = np.array([40.0, 50.0, 60.0])
        _, dist = CoverageEstimator().estimate(depths, use_mle=False)
        assert dist is None

    def test_dist_returned_when_use_mle_true(self):
        rng = np.random.default_rng(0)
        depths = rng.negative_binomial(5, 0.1, size=500).astype(float)
        _, dist = CoverageEstimator().estimate(depths, use_mle=True)
        assert dist is not None


class TestBaseCountFilterFast:

    def _make_vcf_df(self, n_true=30, n_false=20, seed=99):
        rng = np.random.default_rng(seed)
        true_r, true_p = 8.0, 0.12   # mean ~59

        alt_bc_true  = rng.negative_binomial(true_r, true_p, size=n_true)
        ref_bc_true  = rng.integers(0, 5, size=n_true)
        alt_bc_false = rng.integers(0, 5, size=n_false)
        ref_bc_false = rng.negative_binomial(true_r, true_p, size=n_false)

        return (
            pd.DataFrame({
                "alt_bc": np.concatenate([alt_bc_true, alt_bc_false]),
                "ref_bc": np.concatenate([ref_bc_true, ref_bc_false]),
            }),
            sps.nbinom(true_r, true_p)
        )

    def test_true_variants_mostly_positive_ll(self):
        df, dist = self._make_vcf_df()
        filt = BaseCountFilter(power=0.975)
        ll = filt.fit_fast(vcf=df.iloc[:30], distribution=dist)
        assert (ll > 0).sum() > 20

    def test_false_variants_mostly_negative_ll(self):
        df, dist = self._make_vcf_df()
        filt = BaseCountFilter(power=0.975)
        ll = filt.fit_fast(vcf=df.iloc[30:], distribution=dist)
        assert (ll < 0).sum() > 15

    def test_raises_without_distribution_in_fast_mode(self):
        filt = BaseCountFilter()
        df = pd.DataFrame({"alt_bc": [5, 10], "ref_bc": [50, 40]})
        with pytest.raises(ValueError, match="no distribution"):
            filt.fit(
                vcf=df,
                query_coverage_estimate=50.0,
                method="fast",
                distribution=None,
            )

    def test_fit_fast_preserves_index(self):
        rng = np.random.default_rng(0)
        dist = sps.nbinom(5, 0.08)
        df = pd.DataFrame(
            {"alt_bc": rng.integers(0, 60, 10), "ref_bc": rng.integers(0, 60, 10)},
            index=range(100, 110),
        )
        filt = BaseCountFilter(power=0.975)
        ll = filt.fit_fast(vcf=df, distribution=dist)
        assert list(ll.index) == list(df.index)


class TestNucmerSNPFilter:

    def test_snps_in_nucmer_set_marked_fail(self):
        vcf = pd.DataFrame({"pos": [100, 200, 300, 400]})
        nucmer = pd.DataFrame({"pos": [200, 400]})
        result = NucmerSNPFilter().filter(vcf, nucmer)
        assert result.loc[result.pos == 100, "snp_filter"].iloc[0] == "PASS"
        assert result.loc[result.pos == 200, "snp_filter"].iloc[0] == "FAIL"
        assert result.loc[result.pos == 300, "snp_filter"].iloc[0] == "PASS"
        assert result.loc[result.pos == 400, "snp_filter"].iloc[0] == "FAIL"

    def test_empty_nucmer_snps_passes_all(self):
        vcf = pd.DataFrame({"pos": [100, 200, 300]})
        result = NucmerSNPFilter().filter(vcf, pd.DataFrame({"pos": []}))
        assert (result.snp_filter == "PASS").all()


# ---------------------------------------------------------------------------
# Tier 2 — Integration tests (uses session-scoped conftest fixtures)
# ---------------------------------------------------------------------------

class TestFilterCLIFast:

    _base_opts = [
        "--method", "fast", "--variants",
        "-dp", "0", "-a", "5", "-r", "0",
        "-Nc", "500", "-s", "42", "-V", "0",
    ]

    def test_returns_zero(self, synthetic_vcf, synthetic_swp, tmp_path):
        cmd = [
            "cleansweep", "filter",
            str(synthetic_vcf), str(synthetic_swp),
            str(tmp_path / "out"),
        ] + self._base_opts
        rc = subprocess.run(cmd, capture_output=True)
        assert rc.returncode == 0, rc.stderr.decode()

    def test_creates_output_vcf_and_swp(self, synthetic_vcf, synthetic_swp, tmp_path):
        outdir = tmp_path / "out2"
        cmd = [
            "cleansweep", "filter",
            str(synthetic_vcf), str(synthetic_swp),
            str(outdir),
        ] + self._base_opts
        subprocess.run(cmd, capture_output=True, check=True)
        assert (outdir / "cleansweep.variants.vcf").exists()
        assert (outdir / "cleansweep.filter.swp").exists()
