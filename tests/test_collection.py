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
