"""Integration tests for `cleansweep collection` CLI."""
import subprocess


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
