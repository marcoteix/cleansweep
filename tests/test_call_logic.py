"""Unit tests for VariantCaller (cleansweep/call.py) — bcftools path only.

Pilon is not installed in this environment, so `call_pilon`/`method="pilon"`
are intentionally not exercised here.
"""
import subprocess

import pysam
import pytest

import cleansweep.call as call_module
from cleansweep.align import BwaMem
from cleansweep.call import VariantCaller


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:

    def test_non_int_threads_raises(self):
        with pytest.raises(ValueError):
            VariantCaller(threads=1.5)

    def test_non_positive_threads_raises(self):
        with pytest.raises(ValueError):
            VariantCaller(threads=0)

    def test_threads_above_cpu_count_clamped(self, monkeypatch):
        monkeypatch.setattr(call_module, "cpu_count", lambda: 4)
        with pytest.warns(UserWarning):
            vc = VariantCaller(threads=999)
        assert vc.threads == 4


# ---------------------------------------------------------------------------
# TestCallBcftools
# ---------------------------------------------------------------------------

class TestCallBcftools:

    def test_missing_bam_raises(self, synthetic_call_reference, tmp_path):
        vc = VariantCaller(threads=1)
        with pytest.raises(FileNotFoundError):
            vc.call_bcftools(
                bam=tmp_path / "missing.bam",
                reference=synthetic_call_reference,
                output=tmp_path / "out.vcf.gz",
            )

    def test_missing_reference_raises(self, synthetic_call_bam, tmp_path):
        vc = VariantCaller(threads=1)
        with pytest.raises(FileNotFoundError):
            vc.call_bcftools(
                bam=synthetic_call_bam,
                reference=tmp_path / "missing.fa",
                output=tmp_path / "out.vcf.gz",
            )

    def test_creates_valid_vcf(
        self, synthetic_call_bam, synthetic_call_reference, tmp_path
    ):
        vc = VariantCaller(threads=1)
        output = tmp_path / "out.vcf.gz"

        vc.call_bcftools(
            bam=synthetic_call_bam,
            reference=synthetic_call_reference,
            output=output,
        )

        assert output.exists()
        assert output.stat().st_size > 0

        with pysam.VariantFile(str(output)) as vf:
            records = list(vf.fetch())
        assert len(records) >= 1

    def test_mpileup_failure_raises_runtime_error(
        self, synthetic_call_bam, synthetic_call_reference, tmp_path, monkeypatch
    ):
        vc = VariantCaller(threads=1)

        real_run = subprocess.run

        def fake_run(cmd, *args, **kwargs):
            if cmd[:2] == ["bcftools", "mpileup"]:
                return subprocess.CompletedProcess(
                    cmd, returncode=1, stdout=b"", stderr=b"mpileup failed"
                )
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(call_module.subprocess, "run", fake_run)

        with pytest.raises(RuntimeError):
            vc.call_bcftools(
                bam=synthetic_call_bam,
                reference=synthetic_call_reference,
                output=tmp_path / "out.vcf.gz",
            )

    def test_call_step_failure_raises_runtime_error(
        self, synthetic_call_bam, synthetic_call_reference, tmp_path, monkeypatch
    ):
        vc = VariantCaller(threads=1)

        real_run = subprocess.run

        def fake_run(cmd, *args, **kwargs):
            if cmd[:2] == ["bcftools", "call"]:
                return subprocess.CompletedProcess(
                    cmd, returncode=1, stdout=b"", stderr=b"call failed"
                )
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(call_module.subprocess, "run", fake_run)

        with pytest.raises(RuntimeError):
            vc.call_bcftools(
                bam=synthetic_call_bam,
                reference=synthetic_call_reference,
                output=tmp_path / "out.vcf.gz",
            )


# ---------------------------------------------------------------------------
# TestCall — dispatch logic only
# ---------------------------------------------------------------------------

class TestCall:

    def test_unknown_method_raises(
        self, synthetic_call_bam, synthetic_call_reference, tmp_path, monkeypatch
    ):
        vc = VariantCaller(threads=1)
        monkeypatch.setattr(BwaMem, "align", lambda self, reads, reference, output: None)

        with pytest.raises(ValueError):
            vc.call(
                reads=[synthetic_call_bam],
                reference=synthetic_call_reference,
                output=tmp_path / "out.vcf.gz",
                method="unsupported",
            )

    def test_bcftools_method_dispatches_correctly(
        self, synthetic_call_reference, tmp_path, monkeypatch
    ):
        vc = VariantCaller(threads=1)
        output = tmp_path / "out.vcf.gz"
        expected_bam = tmp_path / "alignment.bam"

        monkeypatch.setattr(BwaMem, "align", lambda self, reads, reference, output: None)

        calls = {}

        def fake_call_bcftools(self, bam, reference, output):
            calls["bam"] = bam
            calls["reference"] = reference
            calls["output"] = output

        monkeypatch.setattr(VariantCaller, "call_bcftools", fake_call_bcftools)

        vc.call(
            reads=["reads_1.fq", "reads_2.fq"],
            reference=synthetic_call_reference,
            output=output,
            method="bcftools",
        )

        assert calls["bam"] == expected_bam
        assert calls["reference"] == synthetic_call_reference
        assert calls["output"] == output
