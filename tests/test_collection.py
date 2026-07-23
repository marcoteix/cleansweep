"""Integration tests for `cleansweep collection` CLI."""
import subprocess
from pathlib import Path


def _chrom_line(vcf_path):
    with open(vcf_path) as f:
        for line in f:
            if line.startswith("#CHROM"):
                return line.rstrip("\n")
    raise AssertionError(f"No #CHROM line found in {vcf_path}")


def _sample_names(vcf_path):
    return _chrom_line(vcf_path).split("\t")[9:]


class TestCollectionCLI:

    _base_opts = ["--alpha", "10", "-c", "5"]

    def test_returns_zero(self, synthetic_collection_vcfs, tmp_path):
        vcf_a, vcf_b = synthetic_collection_vcfs
        cmd = [
            "cleansweep", "collection",
            str(vcf_a), str(vcf_b),
            "--output", str(tmp_path / "merged.vcf"),
            "--tmp-dir", str(tmp_path / "tmp"),
        ] + self._base_opts
        rc = subprocess.run(cmd, capture_output=True)
        assert rc.returncode == 0, rc.stderr.decode()

    def test_creates_nonempty_output_vcf(self, synthetic_collection_vcfs, tmp_path):
        vcf_a, vcf_b = synthetic_collection_vcfs
        output = tmp_path / "merged2.vcf"
        cmd = [
            "cleansweep", "collection",
            str(vcf_a), str(vcf_b),
            "--output", str(output),
            "--tmp-dir", str(tmp_path / "tmp2"),
        ] + self._base_opts
        subprocess.run(cmd, capture_output=True, check=True)
        assert output.exists()
        assert output.stat().st_size > 0


class TestCollectionExcludeCLI:

    _base_opts = ["--alpha", "1", "-c", "5"]

    def test_exclude_removes_outlier_sample(
        self, synthetic_collection_vcfs_with_outlier, tmp_path
    ):
        vcf_a, vcf_b, vcf_c = synthetic_collection_vcfs_with_outlier
        name_a, name_b, name_c = (Path(v).stem for v in (vcf_a, vcf_b, vcf_c))
        output = tmp_path / "merged.exclude.vcf"
        cmd = [
            "cleansweep", "collection",
            str(vcf_a), str(vcf_b), str(vcf_c),
            "--output", str(output),
            "--tmp-dir", str(tmp_path / "tmp"),
            "--exclude",
        ] + self._base_opts
        rc = subprocess.run(cmd, capture_output=True)
        assert rc.returncode == 0, rc.stderr.decode()

        samples = _sample_names(output)
        assert name_c not in samples
        assert name_a in samples
        assert name_b in samples

        # Header sample count must match the data rows' column count, or the
        # output VCF is malformed.
        with open(output) as f:
            data_line = next(line for line in f if not line.startswith("#"))
        assert len(data_line.rstrip("\n").split("\t")) == len(_chrom_line(output).split("\t"))

    def test_without_exclude_keeps_all_samples(
        self, synthetic_collection_vcfs_with_outlier, tmp_path
    ):
        vcf_a, vcf_b, vcf_c = synthetic_collection_vcfs_with_outlier
        expected = {Path(v).stem for v in (vcf_a, vcf_b, vcf_c)}
        output = tmp_path / "merged.noexclude.vcf"
        cmd = [
            "cleansweep", "collection",
            str(vcf_a), str(vcf_b), str(vcf_c),
            "--output", str(output),
            "--tmp-dir", str(tmp_path / "tmp2"),
        ] + self._base_opts
        rc = subprocess.run(cmd, capture_output=True)
        assert rc.returncode == 0, rc.stderr.decode()

        samples = _sample_names(output)
        assert set(samples) == expected

    def test_exclude_log_records_excluded_sample(
        self, synthetic_collection_vcfs_with_outlier, tmp_path
    ):
        vcf_a, vcf_b, vcf_c = synthetic_collection_vcfs_with_outlier
        name_c = Path(vcf_c).stem
        exclude_log = tmp_path / "excluded.txt"
        cmd = [
            "cleansweep", "collection",
            str(vcf_a), str(vcf_b), str(vcf_c),
            "--output", str(tmp_path / "merged.log.vcf"),
            "--tmp-dir", str(tmp_path / "tmp3"),
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
            "--output", str(tmp_path / "merged.bad.vcf"),
            "--tmp-dir", str(tmp_path / "tmp4"),
            "--exclude-log", str(tmp_path / "excluded_bad.txt"),
        ] + self._base_opts
        rc = subprocess.run(cmd, capture_output=True)
        assert rc.returncode != 0
