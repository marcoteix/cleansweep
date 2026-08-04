"""Integration tests for `cleansweep collection` CLI — FASTA output."""
import subprocess
from pathlib import Path


def _fasta_names(fasta_path):
    """Return the list of sequence names from a FASTA file (lines starting with '>')."""
    with open(fasta_path) as f:
        return [line.strip()[1:] for line in f if line.startswith(">")]


class TestCollectionCLI:

    _base_opts = ["--alpha", "10", "-c", "5"]

    def test_returns_zero(self, synthetic_collection_vcfs, tmp_path):
        vcf_a, vcf_b = synthetic_collection_vcfs
        cmd = [
            "cleansweep", "collection",
            str(vcf_a), str(vcf_b),
            "--output", str(tmp_path / "out.fasta"),
        ] + self._base_opts
        rc = subprocess.run(cmd, capture_output=True)
        assert rc.returncode == 0, rc.stderr.decode()

    def test_creates_nonempty_output_fasta(self, synthetic_collection_vcfs, tmp_path):
        vcf_a, vcf_b = synthetic_collection_vcfs
        output = tmp_path / "out2.fasta"
        cmd = [
            "cleansweep", "collection",
            str(vcf_a), str(vcf_b),
            "--output", str(output),
        ] + self._base_opts
        subprocess.run(cmd, capture_output=True, check=True)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_output_contains_both_samples(self, synthetic_collection_vcfs, tmp_path):
        vcf_a, vcf_b = synthetic_collection_vcfs
        name_a, name_b = Path(vcf_a).stem, Path(vcf_b).stem
        output = tmp_path / "out3.fasta"
        cmd = [
            "cleansweep", "collection",
            str(vcf_a), str(vcf_b),
            "--output", str(output),
        ] + self._base_opts
        subprocess.run(cmd, capture_output=True, check=True)
        names = _fasta_names(output)
        assert name_a in names
        assert name_b in names


class TestCollectionExcludeCLI:

    _base_opts = ["--alpha", "1", "-c", "5"]

    def test_exclude_removes_outlier_sample(
        self, synthetic_collection_vcfs_with_outlier, tmp_path
    ):
        vcf_a, vcf_b, vcf_c = synthetic_collection_vcfs_with_outlier
        name_a, name_b, name_c = (Path(v).stem for v in (vcf_a, vcf_b, vcf_c))
        output = tmp_path / "exclude.fasta"
        cmd = [
            "cleansweep", "collection",
            str(vcf_a), str(vcf_b), str(vcf_c),
            "--output", str(output),
            "--exclude",
        ] + self._base_opts
        rc = subprocess.run(cmd, capture_output=True)
        assert rc.returncode == 0, rc.stderr.decode()

        names = _fasta_names(output)
        assert name_c not in names
        assert name_a in names
        assert name_b in names

    def test_without_exclude_keeps_all_samples(
        self, synthetic_collection_vcfs_with_outlier, tmp_path
    ):
        vcf_a, vcf_b, vcf_c = synthetic_collection_vcfs_with_outlier
        expected = {Path(v).stem for v in (vcf_a, vcf_b, vcf_c)}
        output = tmp_path / "noexclude.fasta"
        cmd = [
            "cleansweep", "collection",
            str(vcf_a), str(vcf_b), str(vcf_c),
            "--output", str(output),
        ] + self._base_opts
        rc = subprocess.run(cmd, capture_output=True)
        assert rc.returncode == 0, rc.stderr.decode()

        names = _fasta_names(output)
        assert set(names) == expected

    def test_exclude_log_records_excluded_sample(
        self, synthetic_collection_vcfs_with_outlier, tmp_path
    ):
        vcf_a, vcf_b, vcf_c = synthetic_collection_vcfs_with_outlier
        name_c = Path(vcf_c).stem
        exclude_log = tmp_path / "excluded.txt"
        cmd = [
            "cleansweep", "collection",
            str(vcf_a), str(vcf_b), str(vcf_c),
            "--output", str(tmp_path / "log.fasta"),
            "--exclude",
            "--exclude-log", str(exclude_log),
        ] + self._base_opts
        rc = subprocess.run(cmd, capture_output=True)
        assert rc.returncode == 0, rc.stderr.decode()

        assert exclude_log.exists()
        assert exclude_log.read_text().splitlines() == [name_c]

    def test_exclude_log_without_exclude_flag_fails(
        self, synthetic_collection_vcfs_with_outlier, tmp_path
    ):
        vcf_a, vcf_b, vcf_c = synthetic_collection_vcfs_with_outlier
        cmd = [
            "cleansweep", "collection",
            str(vcf_a), str(vcf_b), str(vcf_c),
            "--output", str(tmp_path / "bad.fasta"),
            "--exclude-log", str(tmp_path / "excluded_bad.txt"),
        ] + self._base_opts
        rc = subprocess.run(cmd, capture_output=True)
        assert rc.returncode != 0


def _fasta_records(fasta_path):
    """Return an ordered dict of name -> sequence from a FASTA file."""
    records, name = {}, None
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:]
                records[name] = ""
            elif name is not None:
                records[name] += line
    return records


def _reference_sequence(fasta_path):
    """Return the concatenated sequence of a reference FASTA."""
    return "".join(_fasta_records(fasta_path).values())


class TestCollectionReferenceCLI:
    """The reference-anchored fast path."""

    _base_opts = ["--alpha", "10", "-c", "5"]

    def test_returns_zero(
        self, synthetic_collection_vcfs, synthetic_collection_reference, tmp_path
    ):
        vcf_a, vcf_b = synthetic_collection_vcfs
        cmd = [
            "cleansweep", "collection",
            str(vcf_a), str(vcf_b),
            "--output", str(tmp_path / "ref.fasta"),
            "--reference", str(synthetic_collection_reference),
        ] + self._base_opts
        rc = subprocess.run(cmd, capture_output=True)
        assert rc.returncode == 0, rc.stderr.decode()

    def test_alignment_spans_the_whole_reference(
        self, synthetic_collection_vcfs, synthetic_collection_reference, tmp_path
    ):
        vcf_a, vcf_b = synthetic_collection_vcfs
        output = tmp_path / "span.fasta"
        cmd = [
            "cleansweep", "collection",
            str(vcf_a), str(vcf_b),
            "--output", str(output),
            "--reference", str(synthetic_collection_reference),
        ] + self._base_opts
        subprocess.run(cmd, capture_output=True, check=True)

        expected = len(_reference_sequence(synthetic_collection_reference))
        records = _fasta_records(output)
        assert records, "expected at least one record"
        assert {len(seq) for seq in records.values()} == {expected}

    def test_invariant_positions_take_the_reference_base(
        self, synthetic_collection_vcfs, synthetic_collection_reference, tmp_path
    ):
        """Position 1 has no record in either VCF, so it must be the reference base."""
        vcf_a, vcf_b = synthetic_collection_vcfs
        output = tmp_path / "invariant.fasta"
        cmd = [
            "cleansweep", "collection",
            str(vcf_a), str(vcf_b),
            "--output", str(output),
            "--reference", str(synthetic_collection_reference),
        ] + self._base_opts
        subprocess.run(cmd, capture_output=True, check=True)

        reference = _reference_sequence(synthetic_collection_reference)
        for seq in _fasta_records(output).values():
            assert seq[0] == reference[0]

    def test_gzipped_reference_gives_the_same_result(
        self,
        synthetic_collection_vcfs,
        synthetic_collection_reference,
        synthetic_collection_reference_gz,
        tmp_path,
    ):
        vcf_a, vcf_b = synthetic_collection_vcfs
        outputs = []
        for i, ref in enumerate(
            (synthetic_collection_reference, synthetic_collection_reference_gz)
        ):
            output = tmp_path / f"gz_{i}.fasta"
            cmd = [
                "cleansweep", "collection",
                str(vcf_a), str(vcf_b),
                "--output", str(output),
                "--reference", str(ref),
            ] + self._base_opts
            rc = subprocess.run(cmd, capture_output=True)
            assert rc.returncode == 0, rc.stderr.decode()
            outputs.append(output.read_bytes())

        assert outputs[0] == outputs[1]

    def test_missing_reference_fails(
        self, synthetic_collection_vcfs, tmp_path
    ):
        vcf_a, vcf_b = synthetic_collection_vcfs
        cmd = [
            "cleansweep", "collection",
            str(vcf_a), str(vcf_b),
            "--output", str(tmp_path / "missing.fasta"),
            "--reference", str(tmp_path / "does_not_exist.fa"),
        ] + self._base_opts
        rc = subprocess.run(cmd, capture_output=True)
        assert rc.returncode != 0


class TestCollectionReferenceEquivalence:
    """
    A VCF covering every reference position must give the same alignment whether
    or not a reference FASTA is supplied. This is the load-bearing correctness
    check on the reference-anchored path.
    """

    def _run(self, vcfs, output, reference, alpha):
        cmd = [
            "cleansweep", "collection",
            *[str(v) for v in vcfs],
            "--output", str(output),
            "--alpha", str(alpha),
            "-c", "5",
        ]
        if reference is not None:
            cmd += ["--reference", str(reference)]
        rc = subprocess.run(cmd, capture_output=True)
        assert rc.returncode == 0, rc.stderr.decode()
        return output

    def test_identical_without_outlier_handling(
        self, synthetic_dense_vcfs, synthetic_dense_reference, tmp_path
    ):
        dense = self._run(synthetic_dense_vcfs, tmp_path / "d.fasta", None, 10)
        sparse = self._run(
            synthetic_dense_vcfs, tmp_path / "s.fasta", synthetic_dense_reference, 10
        )
        assert dense.read_bytes() == sparse.read_bytes()

    def test_identical_with_private_snp_removal(
        self, synthetic_dense_vcfs, synthetic_dense_reference, tmp_path
    ):
        """A tiny alpha forces the private-SNP branch to run on both paths."""
        dense = self._run(synthetic_dense_vcfs, tmp_path / "d2.fasta", None, 0.01)
        sparse = self._run(
            synthetic_dense_vcfs, tmp_path / "s2.fasta", synthetic_dense_reference, 0.01
        )
        assert dense.read_bytes() == sparse.read_bytes()

    def test_alignment_matches_the_reference_length(
        self, synthetic_dense_vcfs, synthetic_dense_reference, tmp_path
    ):
        sparse = self._run(
            synthetic_dense_vcfs, tmp_path / "s3.fasta", synthetic_dense_reference, 10
        )
        expected = len(_reference_sequence(synthetic_dense_reference))
        assert {len(s) for s in _fasta_records(sparse).values()} == {expected}

    def test_low_coverage_sites_are_n(
        self, synthetic_dense_vcfs, synthetic_dense_reference, tmp_path
    ):
        sparse = self._run(
            synthetic_dense_vcfs, tmp_path / "s4.fasta", synthetic_dense_reference, 10
        )
        records = _fasta_records(sparse)
        assert any("N" in seq for seq in records.values()), (
            "the dense fixtures include DP=3 sites, which must become N"
        )


class TestCollectionReferenceOutliers:
    """Outlier detection must reach the same verdict on the fast path."""

    _base_opts = ["--alpha", "1", "-c", "5"]

    def test_flags_the_same_outlier(
        self,
        synthetic_collection_vcfs_with_outlier,
        synthetic_collection_reference,
        tmp_path,
    ):
        vcf_a, vcf_b, vcf_c = synthetic_collection_vcfs_with_outlier
        name_a, name_b, name_c = (Path(v).stem for v in (vcf_a, vcf_b, vcf_c))
        exclude_log = tmp_path / "excluded.txt"
        cmd = [
            "cleansweep", "collection",
            str(vcf_a), str(vcf_b), str(vcf_c),
            "--output", str(tmp_path / "out.fasta"),
            "--reference", str(synthetic_collection_reference),
            "--exclude",
            "--exclude-log", str(exclude_log),
        ] + self._base_opts
        rc = subprocess.run(cmd, capture_output=True)
        assert rc.returncode == 0, rc.stderr.decode()

        assert exclude_log.read_text().splitlines() == [name_c]
        names = _fasta_names(tmp_path / "out.fasta")
        assert name_c not in names
        assert name_a in names and name_b in names

    def test_without_exclude_keeps_all_samples(
        self,
        synthetic_collection_vcfs_with_outlier,
        synthetic_collection_reference,
        tmp_path,
    ):
        vcfs = synthetic_collection_vcfs_with_outlier
        expected = {Path(v).stem for v in vcfs}
        output = tmp_path / "keep.fasta"
        cmd = [
            "cleansweep", "collection",
            *[str(v) for v in vcfs],
            "--output", str(output),
            "--reference", str(synthetic_collection_reference),
        ] + self._base_opts
        rc = subprocess.run(cmd, capture_output=True)
        assert rc.returncode == 0, rc.stderr.decode()
        assert set(_fasta_names(output)) == expected


class TestCollectionMultiContig:
    """
    Records are placed by contig as well as position. With the contig ignored,
    two records at the same POS on different contigs collapse onto one column.
    """

    def test_records_map_to_distinct_columns(
        self, synthetic_multicontig_vcfs, synthetic_multicontig_reference, tmp_path
    ):
        vcfs, shared_pos = synthetic_multicontig_vcfs
        output = tmp_path / "multi.fasta"
        cmd = [
            "cleansweep", "collection",
            *[str(v) for v in vcfs],
            "--output", str(output),
            "--reference", str(synthetic_multicontig_reference),
            "--alpha", "10", "-c", "5",
        ]
        rc = subprocess.run(cmd, capture_output=True)
        assert rc.returncode == 0, rc.stderr.decode()

        contigs = _fasta_records(synthetic_multicontig_reference)
        first_length = len(next(iter(contigs.values())))
        reference = "".join(contigs.values())

        for seq in _fasta_records(output).values():
            assert len(seq) == len(reference)
            assert seq[shared_pos - 1] == "T", "alt on the first contig"
            assert seq[first_length + shared_pos - 1] == "T", "alt on the second contig"

    def test_alignment_spans_both_contigs(
        self, synthetic_multicontig_vcfs, synthetic_multicontig_reference, tmp_path
    ):
        vcfs, _ = synthetic_multicontig_vcfs
        output = tmp_path / "multi2.fasta"
        cmd = [
            "cleansweep", "collection",
            *[str(v) for v in vcfs],
            "--output", str(output),
            "--reference", str(synthetic_multicontig_reference),
            "--alpha", "10", "-c", "5",
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        expected = len(_reference_sequence(synthetic_multicontig_reference))
        assert {len(s) for s in _fasta_records(output).values()} == {expected}


class TestCollectionScaling:
    """A collection large enough that the old per-row path would be visible."""

    def test_many_samples(
        self, synthetic_many_vcfs, synthetic_many_reference, tmp_path
    ):
        output = tmp_path / "many.fasta"
        cmd = [
            "cleansweep", "collection",
            *[str(v) for v in synthetic_many_vcfs],
            "--output", str(output),
            "--reference", str(synthetic_many_reference),
            "--alpha", "10", "-c", "5",
            "--n-threads", "4",
        ]
        rc = subprocess.run(cmd, capture_output=True)
        assert rc.returncode == 0, rc.stderr.decode()

        records = _fasta_records(output)
        assert len(records) == len(synthetic_many_vcfs)

        expected = len(_reference_sequence(synthetic_many_reference))
        assert {len(s) for s in records.values()} == {expected}


class TestCollectionOnRealFilterOutput:
    """
    End-to-end over real `cleansweep filter` output. The other fixtures are
    hand-built, so this is the only test that would notice `cleansweep filter`
    changing the FILTER or FORMAT/GT fields that `collection` reads.
    """

    def _run(self, vcfs, output, reference):
        cmd = [
            "cleansweep", "collection",
            *[str(v) for v in vcfs],
            "--output", str(output),
            "--alpha", "10", "-c", "10",
        ]
        if reference is not None:
            cmd += ["--reference", str(reference)]
        rc = subprocess.run(cmd, capture_output=True)
        assert rc.returncode == 0, rc.stderr.decode()
        return output

    def test_both_paths_agree(self, filtered_vcfs, filtered_reference, tmp_path):
        no_ref = self._run(filtered_vcfs, tmp_path / "no_ref.fasta", None)
        with_ref = self._run(filtered_vcfs, tmp_path / "with_ref.fasta", filtered_reference)
        assert no_ref.read_bytes() == with_ref.read_bytes()

    def test_alignment_spans_the_reference(
        self, filtered_vcfs, filtered_reference, tmp_path
    ):
        output = self._run(filtered_vcfs, tmp_path / "span.fasta", filtered_reference)
        expected = len(_reference_sequence(filtered_reference))
        records = _fasta_records(output)
        assert len(records) == len(filtered_vcfs)
        assert {len(s) for s in records.values()} == {expected}

    def test_low_coverage_sites_become_n(
        self, filtered_vcfs, filtered_reference, tmp_path
    ):
        """`cleansweep filter` marks low-depth sites LowCov; those must be N."""
        output = self._run(filtered_vcfs, tmp_path / "lowcov.fasta", filtered_reference)
        for seq in _fasta_records(output).values():
            assert "N" in seq
