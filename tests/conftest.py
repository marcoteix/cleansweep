"""
Pytest fixtures for CleanSweep tests.

All synthetic data is created in a temporary directory shared across the
entire test session (session scope). No fixture reads from real data files.
"""
import subprocess
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pysam
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHROM = "NZ_SYNTHETIC01.1"
CHROM_LEN = 20_000   # large enough for -Nc 500 downsampling
N_VARIANT = 50       # variant positions embedded in the VCF


# ---------------------------------------------------------------------------
# Session-scoped temp directory
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def session_tmpdir():
    with tempfile.TemporaryDirectory(prefix="cleansweep_test_") as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# Synthetic Pilon-format VCF
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def synthetic_vcf(session_tmpdir) -> Path:
    """
    bgzipped, CSI-indexed Pilon-format VCF covering CHROM_LEN positions.

    INFO fields:  DP (total depth), BC (A,C,G,T base counts), MQ (mapping quality)
    FORMAT:       GT
    Depths drawn from NB(5, 0.08) clipped to ≥5 (mean ~57.5).
    ~50 variant positions have high alt_bc; all others are near-ref.
    """
    vcf_path = session_tmpdir / "synthetic.vcf.gz"
    rng = np.random.default_rng(42)

    header = pysam.VariantHeader()
    header.add_line("##fileformat=VCFv4.2")
    header.add_line(f"##contig=<ID={CHROM},length={CHROM_LEN}>")
    header.add_line('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">')
    header.add_line(
        '##INFO=<ID=BC,Number=4,Type=Integer,'
        'Description="Base counts for A,C,G,T">'
    )
    header.add_line('##INFO=<ID=MQ,Number=1,Type=Integer,Description="Mean mapping quality">')
    header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
    header.add_sample("SAMPLE")

    depths = rng.negative_binomial(5, 0.08, size=CHROM_LEN).clip(5).astype(int)
    variant_positions = set(
        rng.choice(range(500, CHROM_LEN - 500), size=N_VARIANT, replace=False)
    )

    with pysam.VariantFile(vcf_path, "wz", header=header) as vf:
        for i in range(CHROM_LEN):
            dp = int(depths[i])
            is_variant = i in variant_positions
            alt_bc = int(rng.integers(dp // 2, dp)) if is_variant else int(rng.integers(0, 5))
            ref_bc = dp - alt_bc

            rec = vf.new_record()
            rec.chrom = CHROM
            rec.pos = i + 1   # pysam write uses 1-based POS
            rec.ref = "A"
            rec.alts = ("T",)
            rec.qual = 60
            rec.info["DP"] = dp
            rec.info["BC"] = (ref_bc, 0, alt_bc, 0)   # A, C, G, T
            rec.info["MQ"] = 60
            rec.samples["SAMPLE"]["GT"] = (1 if is_variant else 0,)
            vf.write(rec)

    rc = subprocess.run(["bcftools", "index", str(vcf_path)], capture_output=True)
    if rc.returncode != 0:
        raise RuntimeError(f"bcftools index failed: {rc.stderr.decode()}")

    return vcf_path


# ---------------------------------------------------------------------------
# Synthetic .swp file (cleansweep prepare output)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def synthetic_swp(session_tmpdir) -> Path:
    """
    joblib-serialised dict mimicking `cleansweep prepare` output.

    gaps: entire chromosome treated as unaligned (end=-1 sentinel).
    snps: empty — no nucmer SNPs between references.
    """
    swp_path = session_tmpdir / "synthetic.swp"

    gaps = pd.DataFrame(
        {"end": [-1]},
        index=pd.Index([0], name="start")
    )

    snps = pd.DataFrame(
        columns=[
            "index", "pos", "ref", "alt", "query_pos",
            "1", "2", "3", "4", "5", "6", "ref_id", "query_id"
        ]
    )

    joblib.dump(
        {"chrom": [CHROM], "gaps": gaps, "snps": snps},
        swp_path,
        compress=3,
    )

    return swp_path


# ---------------------------------------------------------------------------
# Synthetic multi-sample VCFs for collection tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def synthetic_collection_vcfs(session_tmpdir):
    """
    Two bgzipped, CSI-indexed single-sample VCFs for `cleansweep collection`.
    Both share the same contig so bcftools merge can combine them.
    Returns: tuple[Path, Path]
    """
    rng = np.random.default_rng(7)
    vcfs = []

    for sample_name in ("sampleA", "sampleB"):
        vcf_path = session_tmpdir / f"{sample_name}.vcf.gz"

        header = pysam.VariantHeader()
        header.add_line("##fileformat=VCFv4.2")
        header.add_line(f"##contig=<ID={CHROM},length={CHROM_LEN}>")
        header.add_line('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">')
        header.add_line(
            '##INFO=<ID=BC,Number=4,Type=Integer,'
            'Description="Base counts A,C,G,T">'
        )
        header.add_line('##INFO=<ID=MQ,Number=1,Type=Integer,Description="Mean mapping quality">')
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_sample(sample_name)

        variant_positions = sorted(
            rng.choice(range(200, CHROM_LEN - 200), size=20, replace=False)
        )

        with pysam.VariantFile(vcf_path, "wz", header=header) as vf:
            for pos in variant_positions:
                dp = int(rng.integers(30, 80))
                alt_bc = int(rng.integers(dp // 2, dp))
                ref_bc = dp - alt_bc

                rec = vf.new_record()
                rec.chrom = CHROM
                rec.pos = pos
                rec.ref = "A"
                rec.alts = ("T",)
                rec.qual = 60
                rec.info["DP"] = dp
                rec.info["BC"] = (ref_bc, 0, alt_bc, 0)
                rec.info["MQ"] = 60
                rec.samples[sample_name]["GT"] = (1,)
                vf.write(rec)

        rc = subprocess.run(["bcftools", "index", str(vcf_path)], capture_output=True)
        if rc.returncode != 0:
            raise RuntimeError(
                f"bcftools index failed for {vcf_path}: {rc.stderr.decode()}"
            )
        vcfs.append(vcf_path)

    return tuple(vcfs)
