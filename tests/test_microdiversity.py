"""Unit tests for the multiallelic (microdiversity) combination-likelihood engine."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import nbinom

from cleansweep.microdiversity import MicrodiversityModel


def _vcf(rows):
    """Builds a minimal per-allele VCF DataFrame from (ref, A, C, G, T) tuples."""
    df = pd.DataFrame(
        rows,
        columns=["ref", "bc_A", "bc_C", "bc_G", "bc_T"],
    )
    for col in ["bc_A", "bc_C", "bc_G", "bc_T"]:
        df[col] = df[col].astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Combination calling
# ---------------------------------------------------------------------------
class TestEvaluateCalls:

    def setup_method(self):
        self.model = MicrodiversityModel(mu=60.0, alpha=200.0, min_af=0.1)

    def test_monoallelic_reference(self):
        res = self.model.evaluate(_vcf([("A", 58, 1, 0, 2)])).iloc[0]
        assert res.best_combination == frozenset({"A"})
        assert not res.is_multiallelic
        assert res.p_multi == pytest.approx(0.0, abs=1e-6)

    def test_fixed_alternate_allele(self):
        # Reference below threshold, single strong alt -> single non-ref call
        res = self.model.evaluate(_vcf([("A", 2, 0, 0, 57)])).iloc[0]
        assert res.best_combination == frozenset({"T"})
        assert not res.is_multiallelic
        assert res.allele_fractions["T"] == pytest.approx(1.0)

    def test_biallelic_microdiversity(self):
        # Two alleles sharing the coverage ~50/50
        res = self.model.evaluate(_vcf([("A", 30, 0, 28, 0)])).iloc[0]
        assert res.best_combination == frozenset({"A", "G"})
        assert res.is_multiallelic
        assert res.p_multi > 0.9
        assert res.allele_fractions["A"] == pytest.approx(0.5, abs=0.1)
        assert res.allele_fractions["G"] == pytest.approx(0.5, abs=0.1)

    def test_triallelic(self):
        res = self.model.evaluate(_vcf([("A", 0, 20, 18, 16)])).iloc[0]
        assert res.best_combination == frozenset({"C", "G", "T"})
        assert res.is_multiallelic
        assert res.n_candidates == 3

    def test_strong_allele_plus_weak_extra_is_monoallelic(self):
        # One allele already at coverage plus a weak extra: the summed depth would
        # overshoot the coverage, so the extra allele is treated as noise (monoallelic)
        res = self.model.evaluate(_vcf([("A", 58, 0, 12, 0)])).iloc[0]
        assert res.best_combination == frozenset({"A"})
        assert not res.is_multiallelic
        assert res.p_multi < 0.5

    def test_inflated_depth_fails_outlier_gate(self):
        # Total depth far above coverage -> depth-outlier gate flags it
        res = self.model.evaluate(_vcf([("A", 200, 0, 190, 0)])).iloc[0]
        assert not res.evidence_ok

    def test_no_reads_is_empty(self):
        res = self.model.evaluate(_vcf([("A", 0, 0, 0, 0)])).iloc[0]
        assert res.best_combination == frozenset()
        assert not res.evidence_ok

    def test_tiny_counts_fail_outlier_gate(self):
        # A few reads at a site whose strain coverage is ~60: depth-outlier gate fails
        res = self.model.evaluate(_vcf([("A", 3, 2, 1, 0)])).iloc[0]
        assert not res.evidence_ok

    def test_index_preserved(self):
        df = _vcf([("A", 58, 0, 0, 2), ("A", 30, 0, 28, 0)])
        df.index = [100, 205]
        res = self.model.evaluate(df)
        assert list(res.index) == [100, 205]


# ---------------------------------------------------------------------------
# Numerical correctness of the NB-of-sum likelihood
# ---------------------------------------------------------------------------
class TestLikelihood:

    def setup_method(self):
        self.mu, self.alpha = 60.0, 200.0
        self.model = MicrodiversityModel(mu=self.mu, alpha=self.alpha, min_af=0.1)
        self.p_cov = self.alpha / (self.alpha + self.mu)

    def test_combination_logpmf_is_nb_of_sum(self):
        # The likelihood of a combination is NB.logpmf of the summed read count
        for total in (30, 58, 60, 102):
            expected = nbinom.logpmf(total, self.alpha, self.p_cov)
            assert self.model._combination_logpmf(np.array([total]))[0] == pytest.approx(expected)

    def test_best_logL_equals_nb_of_best_sum(self):
        # Biallelic 30/28: best combination is {A,G}, summed depth 58
        res = self.model.evaluate(_vcf([("A", 30, 0, 28, 0)])).iloc[0]
        assert res.best_logL == pytest.approx(nbinom.logpmf(58, self.alpha, self.p_cov))

    def test_p_multi_matches_manual_combination_likelihoods(self):
        # Reconstruct P(multi) for a biallelic site from the NB-of-sum likelihoods
        na, ng = 30, 28
        logs = [
            nbinom.logpmf(na, self.alpha, self.p_cov),          # {A}
            nbinom.logpmf(ng, self.alpha, self.p_cov),          # {G}
            nbinom.logpmf(na + ng, self.alpha, self.p_cov),     # {A,G}
        ]
        expected_p_multi = np.exp(logs[2] - logsumexp(logs))

        result = self.model.evaluate(_vcf([("A", na, 0, ng, 0)])).iloc[0]
        assert result.p_multi == pytest.approx(expected_p_multi, abs=1e-9)

    def test_normalized_combination_probabilities_sum_to_one(self):
        na, ng = 30, 28
        logs = [
            nbinom.logpmf(na, self.alpha, self.p_cov),
            nbinom.logpmf(ng, self.alpha, self.p_cov),
            nbinom.logpmf(na + ng, self.alpha, self.p_cov),
        ]
        probs = np.exp(np.array(logs) - logsumexp(logs))
        assert probs.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Allele fraction recovery
# ---------------------------------------------------------------------------
class TestAlleleFractions:

    def test_fractions_track_observed_ratio(self):
        model = MicrodiversityModel(mu=80.0, alpha=300.0, min_af=0.1)
        # 60/20 split -> 0.75 / 0.25
        res = model.evaluate(_vcf([("A", 60, 0, 20, 0)])).iloc[0]
        assert res.allele_fractions["A"] == pytest.approx(0.75)
        assert res.allele_fractions["G"] == pytest.approx(0.25)
        assert sum(res.allele_fractions.values()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Per-combination minimum allele fraction
# ---------------------------------------------------------------------------
class TestMinAlleleFraction:

    def test_low_fraction_minor_allele_excluded(self):
        # G is 5 / 65 = 0.077 of the {A,G} combination -> below 0.1 -> not multiallelic
        model = MicrodiversityModel(mu=60.0, alpha=200.0, min_af=0.1)
        res = model.evaluate(_vcf([("A", 60, 0, 5, 0)])).iloc[0]
        assert res.best_combination == frozenset({"A"})
        assert not res.is_multiallelic
        assert res.p_multi == pytest.approx(0.0, abs=1e-6)

    def test_lower_min_af_keeps_minor_allele(self):
        # The same site is multiallelic when min_af is below the minor allele fraction
        model = MicrodiversityModel(mu=65.0, alpha=200.0, min_af=0.05)
        res = model.evaluate(_vcf([("A", 60, 0, 5, 0)])).iloc[0]
        assert res.best_combination == frozenset({"A", "G"})
        assert res.is_multiallelic

    def test_high_count_low_fraction_allele_excluded(self):
        # A high-count allele can still be below min_af in a high-coverage sample
        # (30 / 330 = 0.09 < 0.1) -> excluded from the combination
        model = MicrodiversityModel(mu=300.0, alpha=2000.0, min_af=0.1)
        res = model.evaluate(_vcf([("A", 300, 0, 30, 0)])).iloc[0]
        assert res.best_combination == frozenset({"A"})
        assert not res.is_multiallelic
