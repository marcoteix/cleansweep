"""Tests for multiallelic VCF I/O: per-allele parsing, line collapsing, output
expansion, the new INFO fields and the inspect multiallelic statistics."""
import numpy as np
import pandas as pd
import pytest

from cleansweep.vcf import VCF, expand_multiallelic, write_vcf
from cleansweep.inspect import Inspector


# ---------------------------------------------------------------------------
# Per-allele base count parsing
# ---------------------------------------------------------------------------
class TestParseBaseCounts:

    def test_splits_bc_into_four_columns(self):
        vcf = pd.DataFrame({"base_counts": ["30,0,28,2", "10,20,0,0"]})
        out = VCF("x").parse_base_counts(vcf)
        assert list(out["bc_A"]) == [30, 10]
        assert list(out["bc_C"]) == [0, 20]
        assert list(out["bc_G"]) == [28, 0]
        assert list(out["bc_T"]) == [2, 0]

    def test_malformed_bc_becomes_na(self):
        vcf = pd.DataFrame({"base_counts": [None]})
        out = VCF("x").parse_base_counts(vcf)
        assert pd.isna(out["bc_A"].iloc[0])


# ---------------------------------------------------------------------------
# Collapsing caller-split multiallelic lines
# ---------------------------------------------------------------------------
class TestCollapseMultiallelicLines:

    def _df(self):
        # Two records sharing chrom/pos (one ALT per line), site-level BC
        return pd.DataFrame({
            "chrom": ["c1", "c1", "c1"],
            "pos": [100, 200, 200],
            "ref": ["A", "A", "A"],
            "alt": ["T", "T", "C"],
            "base_counts": ["30,0,0,28", "20,18,0,16", "20,18,0,16"],
            "bc_A": pd.array([30, 20, 20], dtype="Int64"),
            "bc_C": pd.array([0, 18, 18], dtype="Int64"),
            "bc_G": pd.array([0, 0, 0], dtype="Int64"),
            "bc_T": pd.array([28, 16, 16], dtype="Int64"),
        })

    def test_collapses_to_one_row_per_site(self):
        out = VCF("x").collapse_multiallelic_lines(self._df())
        assert len(out) == 2

    def test_alt_set_unions_split_alts(self):
        out = VCF("x").collapse_multiallelic_lines(self._df())
        site = out[out.pos == 200].iloc[0]
        assert site["alt_set"] == frozenset({"T", "C"})

    def test_single_line_sites_unchanged(self):
        df = self._df().iloc[[0]]
        out = VCF("x").collapse_multiallelic_lines(df)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# Output expansion
# ---------------------------------------------------------------------------
class TestExpandMultiallelic:

    def _row(self, combination, fractions, ref="A", filt="PASS"):
        return {
            "chrom": "c1", "pos": 100, "id": ".", "ref": ref, "alt": "C",
            "qual": 60, "filter": ".", "info": "DP=60", "format": "GT",
            "sample": "1", "base_counts": "0,30,28,0", "ref_bc": 0, "alt_bc": 30,
            "p_alt": 5.0, "best_combination": combination,
            "allele_fractions": fractions, "best_logL": -7.0, "llr": 3.0,
            "p_multi": 0.95, "is_multiallelic": len(combination) >= 2,
            "cleansweep_filter": filt, "evidence_ok": True,
        }

    def test_two_alts_become_two_rows(self):
        df = pd.DataFrame([self._row(frozenset({"C", "G"}), {"C": 0.52, "G": 0.48})])
        out = expand_multiallelic(df)
        assert len(out) == 2
        assert set(out["alt"]) == {"C", "G"}

    def test_expanded_rows_get_correct_alt_bc_and_fraction(self):
        df = pd.DataFrame([self._row(frozenset({"C", "G"}), {"C": 0.52, "G": 0.48})])
        out = expand_multiallelic(df).set_index("alt")
        assert out.loc["C", "alt_bc"] == 30
        assert out.loc["G", "alt_bc"] == 28
        assert out.loc["C", "allele_fraction"] == pytest.approx(0.52)

    def test_single_nonref_stays_one_row(self):
        df = pd.DataFrame([self._row(frozenset({"A", "G"}), {"A": 0.5, "G": 0.5})])
        out = expand_multiallelic(df)
        assert len(out) == 1
        assert out.iloc[0]["alt"] == "G"

    def test_fail_sites_not_expanded(self):
        df = pd.DataFrame([self._row(frozenset({"C", "G"}), {"C": 0.5, "G": 0.5}, filt="FAIL")])
        out = expand_multiallelic(df)
        assert len(out) == 1

    def test_rows_without_model_columns_pass_through(self):
        df = pd.DataFrame([{
            "chrom": "c1", "pos": 100, "ref": "A", "alt": "T", "filter": "PASS",
        }])
        out = expand_multiallelic(df)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# Round-trip of the new INFO fields through write_vcf -> VCF.read
# ---------------------------------------------------------------------------
class TestInfoRoundTrip:

    _HEADER = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=c1,length=1000>\n"
        '##INFO=<ID=DP,Number=1,Type=Integer,Description="d">\n'
        '##INFO=<ID=BC,Number=4,Type=Integer,Description="bc">\n'
        '##INFO=<ID=MQ,Number=1,Type=Integer,Description="mq">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n"
    )

    def _evaluated_df(self):
        return pd.DataFrame([{
            "chrom": "c1", "pos": 100, "id": ".", "ref": "A", "alt": "G",
            "qual": 60, "filter": ".", "info": "DP=60;BC=30,0,28,0;MQ=60",
            "format": "GT", "sample": "1", "base_counts": "30,0,28,0",
            "ref_bc": 30, "alt_bc": 28, "p_alt": 3.0,
            "best_combination": frozenset({"A", "G"}),
            "allele_fractions": {"A": 0.52, "G": 0.48},
            "best_logL": -7.35, "llr": 2.9, "p_multi": 0.91,
            "is_multiallelic": True, "cleansweep_filter": "PASS", "evidence_ok": True,
        }])

    def test_multiallelic_info_round_trips(self, tmp_path):
        out_file = tmp_path / "out.vcf"
        write_vcf(self._evaluated_df(), out_file, self._HEADER, chrom="c1")

        back = VCF(str(out_file)).read(chrom=None, collapse=False)
        row = back.iloc[0]
        assert row["p_multi"] == pytest.approx(0.91, abs=1e-4)
        assert row["allele_fraction"] == pytest.approx(0.48, abs=1e-4)
        assert row["llr"] == pytest.approx(2.9, abs=1e-4)
        assert bool(row["is_multiallelic"]) is True


# ---------------------------------------------------------------------------
# Inspect: multiallelic count and nucleotide diversity
# ---------------------------------------------------------------------------
class TestInspectMultiallelic:

    def _expanded_output(self):
        # Two multiallelic sites (one ref+alt, one two-alt) plus a monoallelic PASS
        return pd.DataFrame([
            {"chrom": "c1", "pos": 10, "allele_fraction": 0.5, "p_multi": 0.9,
             "is_multiallelic": True},
            {"chrom": "c1", "pos": 20, "allele_fraction": 0.5, "p_multi": 0.8,
             "is_multiallelic": True},
            {"chrom": "c1", "pos": 20, "allele_fraction": 0.5, "p_multi": 0.8,
             "is_multiallelic": True},
            {"chrom": "c1", "pos": 30, "allele_fraction": 1.0, "p_multi": 0.01,
             "is_multiallelic": False},
        ])

    def test_count_multiallelic_sites_counts_distinct_positions(self):
        count = Inspector().count_multiallelic_sites(self._expanded_output())
        assert count == 2

    def test_count_zero_without_column(self):
        assert Inspector().count_multiallelic_sites(pd.DataFrame({"chrom": ["c1"]})) == 0

    def test_nucleotide_diversity_matches_manual(self):
        df = self._expanded_output()
        header = "##contig=<ID=c1,length=1000>\n"
        pi = Inspector().nucleotide_diversity(df, header)

        # Site 10: ref+alt 0.5/0.5 -> 1 - (0.5^2+0.5^2) = 0.5, weighted 0.9
        # Site 20: two alts 0.5/0.5 -> ref_frac 0 -> 1 - (0.5^2+0.5^2) = 0.5, weighted 0.8
        # Site 30: single alt frac 1.0 -> ref_frac 0 -> 1 - 1 = 0
        expected = (0.9 * 0.5 + 0.8 * 0.5 + 0.0) / 1000
        assert pi == pytest.approx(expected)

    def test_nucleotide_diversity_falls_back_without_header(self):
        df = self._expanded_output()
        pi = Inspector().nucleotide_diversity(df, header=None)
        # Fallback denominator = 3 distinct sites
        assert pi == pytest.approx((0.9 * 0.5 + 0.8 * 0.5) / 3)
