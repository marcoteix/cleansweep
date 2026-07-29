"""
Pytest fixtures for CleanSweep tests.

All synthetic data is created in a temporary directory shared across the
entire test session (session scope). No fixture reads from real data files.
"""
import gzip
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


# ---------------------------------------------------------------------------
# Synthetic multi-sample VCFs with one ANI outlier, for --exclude tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def synthetic_collection_vcfs_with_outlier(session_tmpdir):
    """
    Three bgzipped, CSI-indexed single-sample VCFs for `cleansweep collection`.

    sampleA and sampleB share identical genotypes at every site (ANI 1.0 to
    each other). sampleC has an inverted genotype at most sites, giving it a
    much lower ANI to both — it should be flagged as an outlier by the
    ANI-based filter at low --alpha values.

    Returns: tuple[Path, Path, Path] (sampleA, sampleB, sampleC)
    """
    rng = np.random.default_rng(11)
    variant_positions = sorted(
        rng.choice(range(200, CHROM_LEN - 200), size=20, replace=False)
    )

    genotypes = {
        "sampleA": [1] * 20,
        "sampleB": [1] * 20,
        "sampleC": [0] * 18 + [1, 1],
    }

    vcfs = []
    for sample_name, gts in genotypes.items():
        vcf_path = session_tmpdir / f"{sample_name}.outlier.vcf.gz"

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

        with pysam.VariantFile(vcf_path, "wz", header=header) as vf:
            for pos, gt in zip(variant_positions, gts):
                dp = 60
                alt_bc = 45 if gt == 1 else 5
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
                rec.samples[sample_name]["GT"] = (gt,)
                vf.write(rec)

        rc = subprocess.run(["bcftools", "index", str(vcf_path)], capture_output=True)
        if rc.returncode != 0:
            raise RuntimeError(
                f"bcftools index failed for {vcf_path}: {rc.stderr.decode()}"
            )
        vcfs.append(vcf_path)

    return tuple(vcfs)


# ---------------------------------------------------------------------------
# Synthetic FASTA files for prepare tests
# ---------------------------------------------------------------------------

def _write_fasta(path: Path, name: str, seq: str):
    with open(path, "w") as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i + 80] + "\n")


@pytest.fixture(scope="session")
def synthetic_target_fasta(session_tmpdir) -> Path:
    """10 kb synthetic target FASTA."""
    rng = np.random.default_rng(101)
    seq = "".join(rng.choice(list("ACGT"), size=10_000).tolist())
    path = session_tmpdir / "prepare_target.fa"
    _write_fasta(path, "synthetic_target", seq)
    return path


@pytest.fixture(scope="session")
def synthetic_background_fastas(session_tmpdir) -> list:
    """Two 10 kb synthetic background FASTAs (different random seeds)."""
    paths = []
    for i, seed in enumerate([202, 303]):
        rng = np.random.default_rng(seed)
        seq = "".join(rng.choice(list("ACGT"), size=10_000).tolist())
        path = session_tmpdir / f"prepare_background_{i + 1}.fa"
        _write_fasta(path, f"synthetic_background_{i + 1}", seq)
        paths.append(path)
    return paths


@pytest.fixture(scope="session")
def synthetic_target_fasta_gz(session_tmpdir, synthetic_target_fasta) -> Path:
    """Gzipped version of the synthetic target FASTA."""
    gz_path = session_tmpdir / "prepare_target.fa.gz"
    with open(synthetic_target_fasta, "rb") as src, gzip.open(gz_path, "wb") as dst:
        dst.write(src.read())
    return gz_path


@pytest.fixture(scope="session")
def synthetic_background_fastas_gz(session_tmpdir, synthetic_background_fastas) -> list:
    """Gzipped versions of the synthetic background FASTAs."""
    gz_paths = []
    for p in synthetic_background_fastas:
        gz_path = session_tmpdir / (p.name + ".gz")
        with open(p, "rb") as src, gzip.open(gz_path, "wb") as dst:
            dst.write(src.read())
        gz_paths.append(gz_path)
    return gz_paths
