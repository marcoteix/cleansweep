"""Multiallelic (microdiversity) site detection for CleanSweep.

CleanSweep models the depth of coverage of the target strain as a Negative Binomial
distribution. The original model assumes a single allele per site (the query strain
carries either the reference or one alternate allele). This module generalises that to
allow several alleles to co-exist in the query population (microdiversity).

For each site, the per-allele read counts ``(n_A, n_C, n_G, n_T)`` are considered. Every
non-empty combination (``combination``) of the observed alleles is tested. A combination
``S`` is only allowed if every allele in it has an allele fraction of at least
``min_af``, where the allele fraction of an allele within ``S`` is its read count divided
by the combined read count of the alleles in ``S``:

    AF(a, S) = n_a / sum_{i in S} n_i

so an allele whose share of the combination falls below ``min_af`` is treated as
sequencing error within that combination. Microdiversity partitions the target strain's
total depth of coverage among its co-existing alleles, so each allowed combination is
scored by the Negative Binomial likelihood of the *summed* read count of its alleles
under the query coverage distribution:

    log L(S) = NB.logpmf( sum_{i in S} n_i ; mu = coverage, alpha )

The most likely allowed combination is selected, the probability of the site being
multiallelic is

    P(multi) = sum_{|S| >= 2} L(S) / sum_{all S} L(S)   (over allowed combinations)

and the fraction of each allele in the selected combination is its empirical share of the
combined read count.

The model only needs the NB mean (``mu`` = query coverage) and dispersion (``alpha``),
both of which are produced by the fast (MLE) and mixture (MCMC) fitting methods, so the
same engine serves both.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import nbinom

from cleansweep.vcf import _BASES, _BC_COLS

# Clamp value for log-likelihoods that underflow to -inf
_LOG_FLOOR = -1e10


@dataclass
class MicrodiversityModel:
    """Combination-enumeration likelihood for multiallelic sites.

    Args:
        mu (float): Mean depth of coverage of the query strain (NB mean).
        alpha (float): NB dispersion of the query depth of coverage (scipy ``n``).
        min_af (float): Minimum allele fraction, within a combination, for an allele to
            be kept in that combination (combinations with a lower-fraction allele are
            not considered).
        power (float): Central probability mass for the depth-outlier (CDF) gate.
    """

    mu: float
    alpha: float
    min_af: float = 0.1
    power: float = 0.975

    def __post_init__(self):
        self._quantiles = (
            (1 - self.power) / 2,
            (1 + self.power) / 2
        )
        # Negative Binomial success probability for NB(mean=mu, dispersion=alpha)
        self._p_cov = self.alpha / (self.alpha + self.mu)

    def _combination_logpmf(self, summed_counts: np.ndarray) -> np.ndarray:
        """Log-likelihood of a combination from its summed read count under NB(coverage)."""

        return np.nan_to_num(
            nbinom.logpmf(summed_counts, self.alpha, self._p_cov),
            nan=_LOG_FLOOR,
            neginf=_LOG_FLOOR,
            posinf=_LOG_FLOOR
        )

    # --------------------------------------------------------------- public API

    def evaluate(self, vcf: pd.DataFrame) -> pd.DataFrame:
        """Evaluates the multiallelic model for every site in ``vcf``.

        Args:
            vcf (pd.DataFrame): VCF DataFrame with the per-allele counts
                (``bc_A, bc_C, bc_G, bc_T``) and a ``ref`` column.

        Returns:
            pd.DataFrame: Indexed like ``vcf`` with columns ``best_combination``
                (frozenset of allele bases), ``best_logL``, ``llr`` (best vs second-best
                combination log-likelihood ratio), ``p_multi`` (probability the site is
                multiallelic), ``allele_fractions`` (dict base -> fraction),
                ``n_candidates`` (number of observed alleles), ``is_multiallelic`` and
                ``evidence_ok`` (depth-outlier gate).
        """

        n_sites = len(vcf)

        counts = (
            vcf[_BC_COLS]
            .fillna(0)
            .to_numpy(dtype=int)
        )                                           # (n_sites, 4)

        # Alleles observed at each site (any read support). The min_af filter is applied
        # per combination, not as a global candidate threshold.
        present_mask = counts > 0                            # (n_sites, 4)
        n_present = present_mask.sum(axis=1)

        # Pre-allocate outputs
        best_combination = np.empty(n_sites, dtype=object)
        best_logL = np.full(n_sites, np.nan)
        llr = np.full(n_sites, np.nan)
        p_multi = np.zeros(n_sites)
        allele_fractions = np.empty(n_sites, dtype=object)
        evidence_ok = np.ones(n_sites, dtype=bool)

        for p in range(0, 5):
            idx = np.where(n_present == p)[0]
            if len(idx) == 0:
                continue

            if p == 0:
                # No allele has read support: not a callable site
                for i in idx:
                    best_combination[i] = frozenset()
                    allele_fractions[i] = {}
                    evidence_ok[i] = False
                continue

            self._evaluate_group(
                idx=idx,
                p=p,
                counts=counts,
                present_mask=present_mask,
                best_combination=best_combination,
                best_logL=best_logL,
                llr=llr,
                p_multi=p_multi,
                allele_fractions=allele_fractions,
                evidence_ok=evidence_ok,
            )

        is_multiallelic = np.array(
            [
                isinstance(c, frozenset) and len(c) >= 2
                for c in best_combination
            ]
        )

        return pd.DataFrame(
            {
                "best_combination": best_combination,
                "best_logL": best_logL,
                "llr": llr,
                "p_multi": p_multi,
                "allele_fractions": allele_fractions,
                "n_candidates": n_present,
                "is_multiallelic": is_multiallelic,
                "evidence_ok": evidence_ok,
            },
            index=vcf.index,
        )

    # ----------------------------------------------------------- per-k workhorse

    def _evaluate_group(
        self,
        *,
        idx: np.ndarray,
        p: int,
        counts: np.ndarray,
        present_mask: np.ndarray,
        best_combination: np.ndarray,
        best_logL: np.ndarray,
        llr: np.ndarray,
        p_multi: np.ndarray,
        allele_fractions: np.ndarray,
        evidence_ok: np.ndarray,
    ) -> None:
        """Vectorised evaluation of all sites that have exactly ``p`` observed alleles."""

        m = len(idx)

        # Observed allele indices (into A,C,G,T) for each site in the group: (m, p)
        present_positions = np.array(
            [np.flatnonzero(present_mask[i]) for i in idx]
        )
        # Counts and letters of the observed alleles, ordered as in present_positions
        present_counts = np.take_along_axis(counts[idx], present_positions, axis=1)  # (m, p)
        present_letters = np.array(_BASES)[present_positions]                         # (m, p)

        # All non-empty subsets of the p observed alleles
        subsets: List[tuple] = [
            combo
            for r in range(1, p + 1)
            for combo in combinations(range(p), r)
        ]

        # Log-likelihood of every combination for every site. A combination is only
        # allowed when every allele in it has an in-combination fraction >= min_af;
        # disallowed combinations are masked out with -inf so they are never selected
        # and contribute nothing to P(multi).
        sub_logL = np.full((len(subsets), m), -np.inf)
        for j, positions in enumerate(subsets):
            counts_s = present_counts[:, list(positions)]       # (m, s)
            sum_s = counts_s.sum(axis=1)                        # (m,)
            fractions = counts_s / sum_s[:, None]
            allowed = (fractions >= self.min_af).all(axis=1)    # (m,)

            logL = self._combination_logpmf(sum_s)
            sub_logL[j] = np.where(allowed, logL, -np.inf)

        best_j = np.argmax(sub_logL, axis=0)                    # (m,)
        group_best_logL = sub_logL[best_j, np.arange(m)]

        # Number of allowed combinations per site (singletons are always allowed)
        n_allowed = np.isfinite(sub_logL).sum(axis=0)

        # Log-likelihood ratio between the best and second-best allowed combination
        if len(subsets) >= 2:
            second_best = np.partition(sub_logL, -2, axis=0)[-2]
            group_llr = np.where(n_allowed >= 2, group_best_logL - second_best, np.nan)
        else:
            group_llr = np.full(m, np.nan)

        # Probability the site is multiallelic
        multi_rows = [j for j, positions in enumerate(subsets) if len(positions) >= 2]
        if multi_rows:
            log_num = logsumexp(sub_logL[multi_rows], axis=0)
            log_den = logsumexp(sub_logL, axis=0)
            group_p_multi = np.exp(log_num - log_den)
        else:
            group_p_multi = np.zeros(m)

        # Scatter scalar results back to the global arrays
        best_logL[idx] = group_best_logL
        llr[idx] = group_llr
        p_multi[idx] = group_p_multi

        # Resolve the selected combination, allele fractions and depth-outlier gate
        for j in np.unique(best_j):
            positions = subsets[j]
            local = np.where(best_j == j)[0]
            sites = idx[local]

            counts_s = present_counts[local][:, list(positions)]    # (ms, s)

            # Empirical allele fractions among the combined read count
            total_present = counts_s.sum(axis=1)                    # (ms,)
            fractions = counts_s / total_present[:, None]

            # Depth-outlier gate: the summed depth of the present alleles should be
            # consistent with the query coverage distribution
            cdf = nbinom.cdf(total_present, self.alpha, self._p_cov)
            ok = (cdf > self._quantiles[0]) & (cdf < self._quantiles[1])

            for r, site in enumerate(sites):
                letters = [present_letters[local[r], pos] for pos in positions]
                best_combination[site] = frozenset(letters)
                allele_fractions[site] = {
                    letter: float(fractions[r, c])
                    for c, letter in enumerate(letters)
                }
                evidence_ok[site] = bool(ok[r])
