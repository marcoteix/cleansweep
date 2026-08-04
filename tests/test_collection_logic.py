"""Unit tests for Collection logic — no CLI, no multi-process I/O."""
import gzip

import numpy as np
import pytest
from scipy.spatial.distance import hamming

from cleansweep.collection import (
    Collection,
    combine_duplicates,
    load_reference,
    pack_sequences,
)


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
# Fixtures for the reference-anchored path
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chr1_reference(tmp_path_factory):
    """A 3 bp reference for CHR1, the contig `simple_vcf` uses. Bases A, C, G."""
    d = tmp_path_factory.mktemp("chr1_ref")
    p = d / "chr1.fa"
    p.write_text(">CHR1\nACG\n")
    return p


@pytest.fixture(scope="module")
def two_contig_reference(tmp_path_factory):
    """A reference with two 4 bp contigs, for global coordinate mapping."""
    d = tmp_path_factory.mktemp("two_contig")
    p = d / "two.fa"
    p.write_text(">C1\nAAAA\n>C2\nCCCC\n")
    return p


# ---------------------------------------------------------------------------
# TestLoadReference
# ---------------------------------------------------------------------------

class TestLoadReference:

    def test_single_contig_sequence(self, chr1_reference):
        reference, _, _ = load_reference(chr1_reference)
        assert reference.tobytes().decode() == "ACG"

    def test_single_contig_offsets_and_lengths(self, chr1_reference):
        _, offsets, lengths = load_reference(chr1_reference)
        assert offsets == {"CHR1": 0}
        assert lengths == {"CHR1": 3}

    def test_contigs_are_concatenated_in_file_order(self, two_contig_reference):
        reference, offsets, lengths = load_reference(two_contig_reference)
        assert reference.tobytes().decode() == "AAAACCCC"
        assert offsets == {"C1": 0, "C2": 4}
        assert lengths == {"C1": 4, "C2": 4}

    def test_lowercase_is_upcased(self, tmp_path):
        p = tmp_path / "lower.fa"
        p.write_text(">C1\nacgt\n")
        reference, _, _ = load_reference(p)
        assert reference.tobytes().decode() == "ACGT"

    def test_gzipped_reference(self, tmp_path, chr1_reference):
        gz = tmp_path / "chr1.fa.gz"
        with gzip.open(gz, "wt") as f:
            f.write(chr1_reference.read_text())
        reference, offsets, _ = load_reference(gz)
        assert reference.tobytes().decode() == "ACG"
        assert offsets == {"CHR1": 0}

    def test_empty_fasta_raises(self, tmp_path):
        p = tmp_path / "empty.fa"
        p.write_text("")
        with pytest.raises(ValueError):
            load_reference(p)

    def test_duplicate_contig_raises(self, tmp_path):
        p = tmp_path / "dup.fa"
        p.write_text(">C1\nAAAA\n>C1\nCCCC\n")
        with pytest.raises(ValueError):
            load_reference(p)


# ---------------------------------------------------------------------------
# TestVcfToSparse
# ---------------------------------------------------------------------------

class TestVcfToSparse:

    def test_selects_only_non_reference_records(self, col, simple_vcf, chr1_reference):
        """pos 2 is a covered reference call, so it must not be reported."""
        _, offsets, lengths = load_reference(chr1_reference)
        positions, _ = col.vcf_to_sparse(
            vcf=simple_vcf, offsets=offsets, lengths=lengths, min_dp=10
        )
        assert positions.tolist() == [0, 2], "0-based indices of pos 1 and pos 3"

    def test_alt_and_low_coverage_codes(self, col, simple_vcf, chr1_reference):
        _, offsets, lengths = load_reference(chr1_reference)
        _, codes = col.vcf_to_sparse(
            vcf=simple_vcf, offsets=offsets, lengths=lengths, min_dp=10
        )
        assert codes.tobytes().decode() == "TN", "pos 1 is alt T, pos 3 is low coverage"

    def test_unknown_contig_raises(self, col, simple_vcf, two_contig_reference):
        _, offsets, lengths = load_reference(two_contig_reference)
        with pytest.raises(ValueError, match="absent from the reference"):
            col.vcf_to_sparse(
                vcf=simple_vcf, offsets=offsets, lengths=lengths, min_dp=10
            )

    def test_position_past_contig_end_raises(self, col, simple_vcf, tmp_path):
        short = tmp_path / "short.fa"
        short.write_text(">CHR1\nAC\n")
        _, offsets, lengths = load_reference(short)
        with pytest.raises(ValueError, match="outside"):
            col.vcf_to_sparse(
                vcf=simple_vcf, offsets=offsets, lengths=lengths, min_dp=10
            )

    def test_second_contig_is_offset(self, col, tmp_path, two_contig_reference):
        """A record on C2 must land past the end of C1."""
        p = tmp_path / "c2.vcf"
        p.write_text(
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=C1,length=4>\n"
            "##contig=<ID=C2,length=4>\n"
            '##INFO=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n"
            "C1\t2\t.\tA\tT\t60\tPASS\tDP=30\tGT\t1\n"
            "C2\t2\t.\tC\tG\t60\tPASS\tDP=30\tGT\t1\n"
        )
        _, offsets, lengths = load_reference(two_contig_reference)
        positions, codes = col.vcf_to_sparse(
            vcf=p, offsets=offsets, lengths=lengths, min_dp=10
        )
        assert positions.tolist() == [1, 5], "C1:2 -> 1, C2:2 -> 4 + 1"
        assert codes.tobytes().decode() == "TG"


# ---------------------------------------------------------------------------
# TestCombineDuplicates
# ---------------------------------------------------------------------------

class TestCombineDuplicates:

    def _codes(self, s):
        return np.frombuffer(s.encode(), dtype=np.uint8)

    def test_no_duplicates_passes_through(self):
        pos, codes = combine_duplicates(np.array([5, 1, 3]), self._codes("ACG"))
        assert pos.tolist() == [1, 3, 5]
        assert codes.tobytes().decode() == "CGA"

    def test_two_distinct_bases_become_iupac(self):
        pos, codes = combine_duplicates(np.array([7, 7]), self._codes("AG"))
        assert pos.tolist() == [7]
        assert codes.tobytes().decode() == "R", "A + G is IUPAC R"

    def test_iupac_is_order_independent(self):
        _, forward = combine_duplicates(np.array([7, 7]), self._codes("CT"))
        _, reverse = combine_duplicates(np.array([7, 7]), self._codes("TC"))
        assert forward.tobytes() == reverse.tobytes() == b"Y"

    def test_repeated_identical_base_is_kept(self):
        _, codes = combine_duplicates(np.array([2, 2]), self._codes("TT"))
        assert codes.tobytes().decode() == "T"

    def test_any_ambiguous_record_wins(self):
        _, codes = combine_duplicates(np.array([2, 2]), self._codes("AN"))
        assert codes.tobytes().decode() == "N"

    def test_three_bases_become_iupac(self):
        _, codes = combine_duplicates(np.array([1, 1, 1]), self._codes("ACG"))
        assert codes.tobytes().decode() == "V", "A + C + G is IUPAC V"


# ---------------------------------------------------------------------------
# TestMaxIdentities
# ---------------------------------------------------------------------------

class TestMaxIdentities:

    def test_identical_rows_are_fully_identical(self):
        matrix, length = pack_sequences(["ACGT", "ACGT"])
        assert Collection.max_identities(matrix, length).tolist() == [1.0, 1.0]

    def test_single_row_returns_one(self):
        matrix, length = pack_sequences(["ACGT"])
        assert Collection.max_identities(matrix, length).tolist() == [1.0]

    def test_matches_scipy_hamming(self):
        seqs = ["AAAACCCC", "AAAACCCG", "TTTTGGGG"]
        matrix, length = pack_sequences(seqs)
        identities = Collection.max_identities(matrix, length)
        expected = [
            max(1 - hamming(list(a), list(b)) for j, b in enumerate(seqs) if i != j)
            for i, a in enumerate(seqs)
        ]
        assert identities == pytest.approx(expected)

    def test_denominator_is_total_length_not_column_count(self):
        """
        The sparse contract: two rows differing at 1 of 2 held columns, out of a
        1000 bp alignment, are 999/1000 identical -- not 1/2.
        """
        matrix, _ = pack_sequences(["AC", "AG"])
        identities = Collection.max_identities(matrix, total_length=1000)
        assert identities == pytest.approx([0.999, 0.999])


# ---------------------------------------------------------------------------
# TestOutlierIndices
# ---------------------------------------------------------------------------

class TestOutlierIndices:

    def test_fewer_than_two_rows_returns_empty(self):
        matrix, length = pack_sequences(["ACGT"])
        assert Collection.outlier_indices(matrix, length, alpha=1.0) == []

    def test_zero_length_returns_empty(self):
        matrix, _ = pack_sequences(["", ""])
        assert Collection.outlier_indices(matrix, 0, alpha=1.0) == []

    def test_sparse_and_dense_agree(self):
        """
        Dropping the columns where every row agrees must not change the verdict.
        """
        dense = ["A" * 20 + "C", "A" * 20 + "C", "A" * 20 + "G"]
        dense_matrix, dense_length = pack_sequences(dense)
        # Only the last column varies, so that is the whole sparse alignment.
        sparse_matrix, _ = pack_sequences(["C", "C", "G"])
        assert (
            Collection.outlier_indices(sparse_matrix, dense_length, alpha=0.5)
            == Collection.outlier_indices(dense_matrix, dense_length, alpha=0.5)
        )


# ---------------------------------------------------------------------------
# TestConsensusFromMatrix
# ---------------------------------------------------------------------------

class TestConsensusFromMatrix:

    def test_majority_wins(self):
        matrix, _ = pack_sequences(["AATG", "AACG", "AATG"])
        assert Collection.consensus_from_matrix(matrix).tobytes().decode() == "AATG"

    def test_n_is_ignored(self):
        matrix, _ = pack_sequences(["NTG", "ATG", "ATG"])
        assert Collection.consensus_from_matrix(matrix).tobytes().decode() == "ATG"

    def test_all_ambiguous_column_is_n(self):
        matrix, _ = pack_sequences(["N.", "NN"])
        assert Collection.consensus_from_matrix(matrix).tobytes().decode() == "NN"

    def test_tie_resolves_to_smallest_base(self):
        matrix, _ = pack_sequences(["A", "T"])
        assert Collection.consensus_from_matrix(matrix).tobytes().decode() == "A"

    def test_empty_matrix_raises(self):
        with pytest.raises(ValueError):
            Collection.consensus_from_matrix(np.empty((0, 4), dtype=np.uint8))


# ---------------------------------------------------------------------------
# TestRemovePrivateFromMatrix
# ---------------------------------------------------------------------------

class TestRemovePrivateFromMatrix:

    def test_private_base_replaced_with_consensus(self):
        matrix, _ = pack_sequences(["ATCG", "TTCG", "TTCG"])
        consensus = np.frombuffer(b"TTCG", dtype=np.uint8)
        Collection.remove_private_from_matrix(matrix, 0, consensus)
        assert matrix[0].tobytes().decode() == "TTCG"

    def test_shared_base_kept(self):
        matrix, _ = pack_sequences(["ATCG", "ATCG"])
        consensus = np.frombuffer(b"TTTT", dtype=np.uint8)
        Collection.remove_private_from_matrix(matrix, 0, consensus)
        assert matrix[0].tobytes().decode() == "ATCG"

    def test_other_rows_untouched(self):
        matrix, _ = pack_sequences(["ATCG", "TTCG", "TTCG"])
        consensus = np.frombuffer(b"TTCG", dtype=np.uint8)
        Collection.remove_private_from_matrix(matrix, 0, consensus)
        assert matrix[1].tobytes().decode() == "TTCG"
        assert matrix[2].tobytes().decode() == "TTCG"

    def test_single_row_is_a_no_op(self):
        matrix, _ = pack_sequences(["ATCG"])
        Collection.remove_private_from_matrix(matrix, 0, np.frombuffer(b"TTTT", dtype=np.uint8))
        assert matrix[0].tobytes().decode() == "ATCG"

