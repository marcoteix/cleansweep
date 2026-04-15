"""Unit tests for Collection logic — pandas DataFrames only, no file I/O."""
import numpy as np
import pandas as pd
import pytest

from cleansweep.collection import Collection


@pytest.fixture(scope="module")
def col(tmp_path_factory):
    """A Collection instance with stub VCF files to pass __post_init__ checks."""
    d = tmp_path_factory.mktemp("col")
    vcf_a = d / "a.vcf"
    vcf_b = d / "b.vcf"
    vcf_a.touch()
    vcf_b.touch()
    return Collection(
        vcfs=[vcf_a, vcf_b],
        output=d / "out.vcf",
        tmp_dir=d / "tmp",
        alpha=10.0,
    )


class TestSnpDistance:

    def test_identical_series_returns_zero(self, col):
        s = pd.Series(["1", "0", "1", "0", "."])
        assert col.snp_distance(s, s) == 0

    def test_missing_positions_excluded(self, col):
        s1 = pd.Series(["1", ".", "0"])
        s2 = pd.Series(["0", "0", "0"])
        # pos 0 differs, pos 1 skipped (s1='.'), pos 2 same
        assert col.snp_distance(s1, s2) == 1

    def test_both_missing_excluded(self, col):
        s1 = pd.Series([".", "1"])
        s2 = pd.Series([".", "0"])
        assert col.snp_distance(s1, s2) == 1

    def test_all_missing_returns_zero(self, col):
        s1 = pd.Series([".", "."])
        s2 = pd.Series([".", "."])
        assert col.snp_distance(s1, s2) == 0

    def test_known_difference_count(self, col):
        s1 = pd.Series(["1", "0", "1", "1", "0"])
        s2 = pd.Series(["1", "0", "0", "0", "1"])
        assert col.snp_distance(s1, s2) == 3


class TestSnpMatrix:

    def test_diagonal_is_zero(self, col):
        geno = pd.DataFrame({"sA": ["1", "0", "1"], "sB": ["0", "1", "0"]})
        mat = col.snp_matrix(geno)
        assert mat.loc["sA", "sA"] == 0
        assert mat.loc["sB", "sB"] == 0

    def test_symmetric(self, col):
        geno = pd.DataFrame({
            "sA": ["1", "0", "1"],
            "sB": ["0", "1", "1"],
            "sC": ["0", "0", "1"],
        })
        mat = col.snp_matrix(geno)
        assert mat.loc["sA", "sB"] == mat.loc["sB", "sA"]
        assert mat.loc["sA", "sC"] == mat.loc["sC", "sA"]

    def test_known_distances(self, col):
        geno = pd.DataFrame({"sA": ["1", "0"], "sB": ["0", "1"]})
        mat = col.snp_matrix(geno)
        assert mat.loc["sA", "sB"] == 2

    def test_column_and_index_names(self, col):
        geno = pd.DataFrame({"x": ["1"], "y": ["0"], "z": ["1"]})
        mat = col.snp_matrix(geno)
        assert list(mat.columns) == ["x", "y", "z"]
        assert list(mat.index) == ["x", "y", "z"]


class TestCoreSnps:

    def test_private_snp_not_core(self, col):
        """SNP in only 1 of 3 samples → not core."""
        geno = pd.DataFrame({"sA": ["1"], "sB": ["0"], "sC": ["0"]})
        _, mask = col.core_snps(geno)
        assert not mask.iloc[0]

    def test_all_samples_alt_is_core(self, col):
        """All samples have alt → n_samples == n_pass → core."""
        geno = pd.DataFrame({"sA": ["1"], "sB": ["1"], "sC": ["1"]})
        _, mask = col.core_snps(geno)
        assert mask.iloc[0]

    def test_single_passing_sample_is_not_core(self, col):
        """Only 1 sample has data (n_pass == 1) → not core."""
        geno = pd.DataFrame({"sA": ["1"], "sB": ["."], "sC": ["."]})
        _, mask = col.core_snps(geno)
        assert not mask.iloc[0]

    def test_snp_in_two_of_five_is_core(self, col):
        """2/5 samples have alt → 1 < n_samples < n_pass-1 → core."""
        geno = pd.DataFrame({
            "s1": ["1"], "s2": ["1"], "s3": ["0"], "s4": ["0"], "s5": ["0"]
        })
        _, mask = col.core_snps(geno)
        assert mask.iloc[0]

    def test_all_ref_is_core(self, col):
        """n_samples == 0 → core."""
        geno = pd.DataFrame({"sA": ["0"], "sB": ["0"]})
        _, mask = col.core_snps(geno)
        assert mask.iloc[0]


class TestOutlierDetectionLogic:
    """
    Test the IQR-based outlier detection logic directly using helper methods,
    without reading any VCF files.
    """

    def test_uniform_samples_iqr_nonnegative(self, col):
        """All pairwise ANIs equal → IQR = 0, no crash."""
        geno = pd.DataFrame({
            "sA": ["1", "0", "1"],
            "sB": ["1", "0", "0"],
            "sC": ["1", "0", "0"],
        })
        snp_mat = col.snp_matrix(geno)
        ani_matrix = 1.0 - snp_mat / 10_000
        n = len(ani_matrix)
        pairwise = ani_matrix.values[np.triu_indices(n, k=1)]
        iqr = float(np.percentile(pairwise, 75) - np.percentile(pairwise, 25))
        assert iqr >= 0

    def test_divergent_sample_falls_below_threshold(self, col):
        """
        10 identical samples + 1 fully divergent sample.
        With alpha=1.0 the divergent sample's max_ani should be < threshold.
        """
        n_close = 10
        n_pos = 20
        base = ["1" if i % 2 == 0 else "0" for i in range(n_pos)]
        inv  = ["0" if x == "1" else "1" for x in base]

        data = {f"s{i}": base[:] for i in range(n_close)}
        data["sK"] = inv
        geno = pd.DataFrame(data)

        genome_len = 10_000
        snp_mat = col.snp_matrix(geno)
        ani_matrix = 1.0 - snp_mat / genome_len

        n = len(ani_matrix)
        pairwise = ani_matrix.values[np.triu_indices(n, k=1)]
        ani_median = float(np.median(pairwise))
        ani_iqr = float(np.percentile(pairwise, 75) - np.percentile(pairwise, 25))
        threshold = ani_median - 1.0 * ani_iqr

        max_ani_sK = float(ani_matrix.loc["sK"].drop("sK").max())
        assert max_ani_sK < threshold, (
            f"Expected sK (max_ani={max_ani_sK:.6f}) to be below "
            f"threshold={threshold:.6f}"
        )

    def test_similar_samples_not_flagged_with_large_alpha(self, col):
        """With alpha=10 no sample should be filtered when all are similar."""
        n_pos = 10
        rng = np.random.default_rng(42)
        data = {
            f"s{i}": [str(rng.integers(0, 2)) for _ in range(n_pos)]
            for i in range(5)
        }
        geno = pd.DataFrame(data)

        genome_len = 50_000
        snp_mat = col.snp_matrix(geno)
        ani_matrix = 1.0 - snp_mat / genome_len

        n = len(ani_matrix)
        pairwise = ani_matrix.values[np.triu_indices(n, k=1)]
        ani_median = float(np.median(pairwise))
        ani_iqr = float(np.percentile(pairwise, 75) - np.percentile(pairwise, 25))
        threshold = ani_median - 10.0 * ani_iqr

        for sample in ani_matrix.index:
            max_ani = float(ani_matrix.loc[sample].drop(sample).max())
            assert max_ani >= threshold, (
                f"Sample {sample} was unexpectedly flagged with alpha=10"
            )
