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


# ---------------------------------------------------------------------------
# Reference FASTA matching the contig used by the collection VCF fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def synthetic_collection_reference(session_tmpdir) -> Path:
    """
    Reference FASTA for CHROM, the contig `synthetic_collection_vcfs` and
    `synthetic_collection_vcfs_with_outlier` place their records on. Lets those
    fixtures be run through the reference-anchored path of `collection`.
    """
    rng = np.random.default_rng(909)
    seq = "".join(rng.choice(list("ACGT"), size=CHROM_LEN).tolist())
    path = session_tmpdir / "collection_reference.fa"
    _write_fasta(path, CHROM, seq)
    return path


@pytest.fixture(scope="session")
def synthetic_collection_reference_gz(session_tmpdir, synthetic_collection_reference) -> Path:
    """Gzipped version of the collection reference FASTA."""
    gz_path = session_tmpdir / "collection_reference.fa.gz"
    with open(synthetic_collection_reference, "rb") as src, gzip.open(gz_path, "wb") as dst:
        dst.write(src.read())
    return gz_path


# ---------------------------------------------------------------------------
# Dense VCFs consistent with their reference, for old-path/new-path equivalence
# ---------------------------------------------------------------------------
DENSE_CHROM = "NZ_DENSE01.1"
DENSE_LEN = 2_000


@pytest.fixture(scope="session")
def synthetic_dense_reference(session_tmpdir) -> Path:
    """A 2 kb reference FASTA for the dense collection fixtures."""
    rng = np.random.default_rng(4242)
    seq = "".join(rng.choice(list("ACGT"), size=DENSE_LEN).tolist())
    path = session_tmpdir / "dense_reference.fa"
    _write_fasta(path, DENSE_CHROM, seq)
    return path


@pytest.fixture(scope="session")
def synthetic_dense_vcfs(session_tmpdir, synthetic_dense_reference):
    """
    Three bgzipped, CSI-indexed VCFs with a record at *every* reference
    position, with REF taken from `synthetic_dense_reference`.

    Because they cover every site and agree with the reference, converting them
    with and without a reference FASTA must give byte-identical alignments —
    which is the load-bearing check on the reference-anchored fast path.

    Each sample gets its own variant positions, its own low-coverage positions
    (DP=3, below the -c 5 the tests use), and one missing genotype, so every
    branch of the base classification is exercised.

    Returns: tuple[Path, Path, Path]
    """
    ref_seq = "".join(
        line.strip()
        for line in synthetic_dense_reference.read_text().splitlines()[1:]
    )

    rng = np.random.default_rng(31337)
    vcfs = []

    for offset, sample_name in enumerate(("dense1", "dense2", "dense3")):
        vcf_path = session_tmpdir / f"{sample_name}.vcf.gz"

        variants = set(rng.choice(range(1, DENSE_LEN + 1), size=40, replace=False).tolist())
        low_cov = set(rng.choice(range(1, DENSE_LEN + 1), size=25, replace=False).tolist())
        missing = {17 + offset}

        header = pysam.VariantHeader()
        header.add_line("##fileformat=VCFv4.2")
        header.add_line(f"##contig=<ID={DENSE_CHROM},length={DENSE_LEN}>")
        header.add_line('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">')
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_sample(sample_name)

        with pysam.VariantFile(vcf_path, "wz", header=header) as vf:
            for pos in range(1, DENSE_LEN + 1):
                ref_base = ref_seq[pos - 1]
                # A deterministic alternate allele that is never the reference.
                alt_base = "ACGT"[("ACGT".index(ref_base) + 1) % 4]

                rec = vf.new_record()
                rec.chrom = DENSE_CHROM
                rec.pos = pos
                rec.ref = ref_base
                rec.alts = (alt_base,)
                rec.qual = 60
                rec.info["DP"] = 3 if pos in low_cov else 30

                if pos in missing:
                    rec.samples[sample_name]["GT"] = (None,)
                else:
                    rec.samples[sample_name]["GT"] = (1 if pos in variants else 0,)

                vf.write(rec)

        rc = subprocess.run(["bcftools", "index", str(vcf_path)], capture_output=True)
        if rc.returncode != 0:
            raise RuntimeError(
                f"bcftools index failed for {vcf_path}: {rc.stderr.decode()}"
            )
        vcfs.append(vcf_path)

    return tuple(vcfs)


# ---------------------------------------------------------------------------
# Multi-contig reference and VCF
# ---------------------------------------------------------------------------
MULTI_CONTIGS = (("NZ_MULTI01.1", 300), ("NZ_MULTI02.1", 200))


@pytest.fixture(scope="session")
def synthetic_multicontig_reference(session_tmpdir) -> Path:
    """A two-contig reference FASTA, for testing global coordinate mapping."""
    rng = np.random.default_rng(5150)
    path = session_tmpdir / "multicontig_reference.fa"

    with open(path, "w") as f:
        for name, length in MULTI_CONTIGS:
            seq = "".join(rng.choice(list("ACGT"), size=length).tolist())
            f.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")

    return path


@pytest.fixture(scope="session")
def synthetic_multicontig_vcfs(session_tmpdir, synthetic_multicontig_reference):
    """
    Two bgzipped, CSI-indexed VCFs with an alternate call on each of the two
    contigs of `synthetic_multicontig_reference`, at the same POS on both.

    The shared POS is the point: with the contig ignored, the two records
    collapse onto one alignment column.

    Returns: tuple[tuple[Path, Path], int]  -- the VCFs and the shared POS.
    """
    shared_pos = 50
    records = {}

    for name, _ in MULTI_CONTIGS:
        records[name] = shared_pos

    vcfs = []
    for sample_name in ("multi1", "multi2"):
        vcf_path = session_tmpdir / f"{sample_name}.multi.vcf.gz"

        header = pysam.VariantHeader()
        header.add_line("##fileformat=VCFv4.2")
        for name, length in MULTI_CONTIGS:
            header.add_line(f"##contig=<ID={name},length={length}>")
        header.add_line('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">')
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_sample(sample_name)

        with pysam.VariantFile(vcf_path, "wz", header=header) as vf:
            for chrom, pos in records.items():
                rec = vf.new_record()
                rec.chrom = chrom
                rec.pos = pos
                rec.ref = "A"
                rec.alts = ("T",)
                rec.qual = 60
                rec.info["DP"] = 30
                rec.samples[sample_name]["GT"] = (1,)
                vf.write(rec)

        rc = subprocess.run(["bcftools", "index", str(vcf_path)], capture_output=True)
        if rc.returncode != 0:
            raise RuntimeError(
                f"bcftools index failed for {vcf_path}: {rc.stderr.decode()}"
            )
        vcfs.append(vcf_path)

    return tuple(vcfs), shared_pos


# ---------------------------------------------------------------------------
# A larger collection, for the scaling smoke test
# ---------------------------------------------------------------------------
MANY_CHROM = "NZ_MANY01.1"
MANY_LEN = 50_000
MANY_SAMPLES = 40


@pytest.fixture(scope="session")
def synthetic_many_reference(session_tmpdir) -> Path:
    """A 50 kb reference for the scaling fixtures."""
    rng = np.random.default_rng(2718)
    seq = "".join(rng.choice(list("ACGT"), size=MANY_LEN).tolist())
    path = session_tmpdir / "many_reference.fa"
    _write_fasta(path, MANY_CHROM, seq)
    return path


@pytest.fixture(scope="session")
def synthetic_many_vcfs(session_tmpdir, synthetic_many_reference):
    """
    Forty sparse, bgzipped, CSI-indexed VCFs over a 50 kb reference.

    Only variant and low-coverage sites are recorded, which is what a real
    `cleansweep filter --variants` run produces, so this exercises the fast path
    at a sample count where the old per-row conversion would dominate.

    Returns: tuple[Path, ...]
    """
    ref_seq = "".join(
        line.strip()
        for line in synthetic_many_reference.read_text().splitlines()[1:]
    )

    rng = np.random.default_rng(161803)
    # A shared core of variants plus a per-sample tail, so the samples are
    # related without being identical.
    core = sorted(rng.choice(range(1, MANY_LEN + 1), size=150, replace=False).tolist())

    vcfs = []
    for i in range(MANY_SAMPLES):
        vcf_path = session_tmpdir / f"many{i:03d}.vcf.gz"

        private = rng.choice(range(1, MANY_LEN + 1), size=30, replace=False).tolist()
        low_cov = set(rng.choice(range(1, MANY_LEN + 1), size=20, replace=False).tolist())
        positions = sorted(set(core) | set(private) | low_cov)

        header = pysam.VariantHeader()
        header.add_line("##fileformat=VCFv4.2")
        header.add_line(f"##contig=<ID={MANY_CHROM},length={MANY_LEN}>")
        header.add_line('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">')
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_sample(f"many{i:03d}")

        with pysam.VariantFile(vcf_path, "wz", header=header) as vf:
            for pos in positions:
                ref_base = ref_seq[pos - 1]
                alt_base = "ACGT"[("ACGT".index(ref_base) + 1) % 4]

                rec = vf.new_record()
                rec.chrom = MANY_CHROM
                rec.pos = pos
                rec.ref = ref_base
                rec.alts = (alt_base,)
                rec.qual = 60
                rec.info["DP"] = 3 if pos in low_cov else 40
                rec.samples[f"many{i:03d}"]["GT"] = (0 if pos in low_cov else 1,)
                vf.write(rec)

        rc = subprocess.run(["bcftools", "index", str(vcf_path)], capture_output=True)
        if rc.returncode != 0:
            raise RuntimeError(
                f"bcftools index failed for {vcf_path}: {rc.stderr.decode()}"
            )
        vcfs.append(vcf_path)

    return tuple(vcfs)


# ---------------------------------------------------------------------------
# Real `cleansweep filter` output, for an end-to-end collection test
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def filtered_vcfs(session_tmpdir, synthetic_vcf, synthetic_swp):
    """
    Three real `cleansweep filter` output VCFs.

    Every other collection fixture is hand-built, so nothing else would catch a
    change in how `cleansweep filter` writes its FILTER or FORMAT/GT fields —
    which is exactly what `collection` reads.

    Returns: tuple[Path, Path, Path]
    """
    outdir = session_tmpdir / "filtered"
    outdir.mkdir(exist_ok=True)

    vcfs = []
    for name in ("f1", "f2", "f3"):
        rc = subprocess.run(
            [
                "cleansweep", "filter",
                str(synthetic_vcf), str(synthetic_swp), str(outdir / name),
                "--method", "fast",
                "-dp", "5", "-a", "5", "-r", "0", "-Nc", "500", "-s", "42", "-V", "0",
            ],
            capture_output=True,
        )
        if rc.returncode != 0:
            raise RuntimeError(f"cleansweep filter failed: {rc.stderr.decode()}")

        produced = outdir / name / "cleansweep.variants.vcf"
        if not produced.exists():
            raise RuntimeError(f"cleansweep filter wrote no VCF for {name}")

        # Give each one a distinct stem, since collection names records after it.
        renamed = outdir / f"{name}.vcf"
        renamed.write_bytes(produced.read_bytes())
        vcfs.append(renamed)

    return tuple(vcfs)


@pytest.fixture(scope="session")
def filtered_reference(session_tmpdir) -> Path:
    """
    Reference FASTA matching `filtered_vcfs`.

    `synthetic_vcf` uses REF="A" at every position, so the reference has to as
    well for the two conversion paths to be comparable.
    """
    path = session_tmpdir / "filtered_reference.fa"
    _write_fasta(path, CHROM, "A" * CHROM_LEN)
    return path
