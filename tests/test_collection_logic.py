"""Unit tests for Collection logic — no CLI, no multi-process I/O."""
import numpy as np
import pytest

from cleansweep.collection import Collection


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

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
        output=d / "out.fasta",
        alpha=10.0,
        n_threads=1,
    )


# ---------------------------------------------------------------------------
# Minimal VCF fixture for vcf_to_seq tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def simple_vcf(tmp_path_factory):
    """
    Plain-text VCF with 3 SNP positions:
      pos 1 → alt (T), depth 30
      pos 2 → ref (C), depth 30
      pos 3 → alt (T), depth 5  (below min_dp=10 → N)
    Expected sequence: "TCN"
    """
    d = tmp_path_factory.mktemp("vcf_seq")
    p = d / "simple.vcf"
    p.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=CHR1,length=3>\n"
        '##INFO=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
        '##INFO=<ID=BC,Number=4,Type=Integer,Description="Base counts A,C,G,T">\n'
        '##INFO=<ID=MQ,Number=1,Type=Integer,Description="Mapping quality">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n"
        "CHR1\t1\t.\tA\tT\t60\tPASS\tDP=30;BC=0,0,0,30;MQ=60\tGT\t1\n"
        "CHR1\t2\t.\tC\tG\t60\tPASS\tDP=30;BC=30,0,0,0;MQ=60\tGT\t0\n"
        "CHR1\t3\t.\tA\tT\t60\tPASS\tDP=5;BC=0,0,0,5;MQ=60\tGT\t1\n"
    )
    return p


# ---------------------------------------------------------------------------
# TestVcfToSeq
# ---------------------------------------------------------------------------

class TestVcfToSeq:

    def test_alt_base_written(self, col, simple_vcf):
        seq = col.vcf_to_seq(vcf=simple_vcf, min_dp=10)
        assert seq[0] == "T", "pos 1 has GT=1 (alt=T)"

    def test_ref_base_written(self, col, simple_vcf):
        seq = col.vcf_to_seq(vcf=simple_vcf, min_dp=10)
        assert seq[1] == "C", "pos 2 has GT=0 (ref=C)"

    def test_low_depth_written_as_n(self, col, simple_vcf):
        seq = col.vcf_to_seq(vcf=simple_vcf, min_dp=10)
        assert seq[2] == "N", "pos 3 has depth 5 < min_dp=10"

    def test_full_sequence(self, col, simple_vcf):
        assert col.vcf_to_seq(vcf=simple_vcf, min_dp=10) == "TCN"

    def test_sequence_length_equals_last_pos(self, col, simple_vcf):
        seq = col.vcf_to_seq(vcf=simple_vcf, min_dp=10)
        assert len(seq) == 3

    def test_named_sample_column_works(self, col, tmp_path_factory):
        """vcf_to_seq must work when the sample column is not literally 'SAMPLE'."""
        d = tmp_path_factory.mktemp("named_sample")
        p = d / "named.vcf"
        p.write_text(
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=CHR1,length=2>\n"
            '##INFO=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
            '##INFO=<ID=BC,Number=4,Type=Integer,Description="Base counts A,C,G,T">\n'
            '##INFO=<ID=MQ,Number=1,Type=Integer,Description="Mapping quality">\n'
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tmySample\n"
            "CHR1\t1\t.\tA\tT\t60\tPASS\tDP=30;BC=0,0,0,30;MQ=60\tGT\t1\n"
            "CHR1\t2\t.\tC\tG\t60\tPASS\tDP=30;BC=30,0,0,0;MQ=60\tGT\t0\n"
        )
        seq = col.vcf_to_seq(vcf=p, min_dp=10)
        assert seq == "TC"


# ---------------------------------------------------------------------------
# TestFindOutliers
# ---------------------------------------------------------------------------

class TestFindOutliers:

    def test_single_sequence_returns_empty(self, col):
        assert col.find_outliers({"s1": "AAAA"}) == []

    def test_two_identical_no_outlier(self, col):
        seqs = {"s1": "AAAA", "s2": "AAAA"}
        assert col.find_outliers(seqs, alpha=1.0) == []

    def test_divergent_sample_flagged(self, col):
        similar = "A" * 50
        outlier = "T" * 50
        seqs = {f"s{i}": similar for i in range(4)}
        seqs["sOUT"] = outlier
        result = col.find_outliers(seqs, alpha=1.0)
        assert "sOUT" in result
        assert all(f"s{i}" not in result for i in range(4))

    def test_all_similar_large_alpha_no_outlier(self, col):
        rng = np.random.default_rng(42)
        seqs = {
            f"s{i}": "".join(rng.choice(list("ACGT"), size=40).tolist())
            for i in range(5)
        }
        result = col.find_outliers(seqs, alpha=10.0)
        assert result == []


# ---------------------------------------------------------------------------
# TestConsensusSequence
# ---------------------------------------------------------------------------

class TestConsensusSequence:

    def test_unanimous_position(self, col):
        seqs = {"s1": "AAA", "s2": "AAA", "s3": "AAA"}
        cons = col.consensus_sequence(seqs)
        assert list(cons) == ["A", "A", "A"]

    def test_majority_wins(self, col):
        seqs = {"s1": "AATG", "s2": "AACG", "s3": "AATG"}
        cons = col.consensus_sequence(seqs)
        assert cons[2] == "T"

    def test_n_ignored_in_consensus(self, col):
        seqs = {"s1": "NTG", "s2": "ATG", "s3": "ATG"}
        cons = col.consensus_sequence(seqs)
        assert cons[0] == "A"

    def test_empty_sequences_raises(self, col):
        with pytest.raises(ValueError):
            col.consensus_sequence({})

    def test_output_length_matches_sequence_length(self, col):
        seqs = {"s1": "ACGT", "s2": "ACGT"}
        assert len(col.consensus_sequence(seqs)) == 4


# ---------------------------------------------------------------------------
# TestRemovePrivateSnps
# ---------------------------------------------------------------------------

class TestRemovePrivateSnps:

    def test_private_snp_replaced_with_consensus(self, col):
        target = "ATCG"
        others = {"s2": "TTCG", "s3": "TTCG"}
        consensus = np.array(["T", "T", "C", "G"])
        result = col.remove_private_snps(target, others, consensus)
        assert result[0] == "T", "private A → replaced with consensus T"

    def test_shared_snp_kept(self, col):
        target = "ATCG"
        others = {"s2": "ATCG"}
        consensus = np.array(["A", "T", "C", "G"])
        result = col.remove_private_snps(target, others, consensus)
        assert result == "ATCG"

    def test_output_same_length_as_input(self, col):
        target = "AAAA"
        others = {"s2": "TTTT"}
        consensus = np.array(["T", "T", "T", "T"])
        result = col.remove_private_snps(target, others, consensus)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# TestWriteMsa
# ---------------------------------------------------------------------------

class TestWriteMsa:

    def test_fasta_format(self, col, tmp_path):
        seqs = {"s1": "ATCG", "s2": "ATCG"}
        out = tmp_path / "out.fasta"
        col.write_msa(seqs, out)
        content = out.read_text()
        assert content == ">s1\nATCG\n>s2\nATCG\n"

    def test_raises_on_different_lengths(self, col, tmp_path):
        with pytest.raises(ValueError):
            col.write_msa({"s1": "AT", "s2": "ATCG"}, tmp_path / "bad.fasta")

    def test_creates_file(self, col, tmp_path):
        out = tmp_path / "out2.fasta"
        col.write_msa({"s1": "ACGT"}, out)
        assert out.exists() and out.stat().st_size > 0
