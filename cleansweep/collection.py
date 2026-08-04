#%%
"""
Builds a multiple sequence alignment from a collection of CleanSweep VCFs.

The alignment is carried internally as a ``samples x positions`` array of base
codes rather than as full-length strings. When a reference FASTA is supplied,
only positions that resolve to something other than the reference base in at
least one sample are held in that array, which keeps every stage of the
pipeline proportional to the number of variant and low-coverage sites instead
of to the genome length.

Author: Marco Teixeira
Email: mcarvalh@broadinstitute.org
"""
import gzip
import logging
import warnings
from dataclasses import dataclass
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from Bio import SeqIO

from cleansweep.typing import File
from cleansweep.vcf import IUPAC_CODES, VCF, get_info_value

# Bases are held as ASCII byte codes, so the alignment costs one byte per cell.
_N_CODE = np.uint8(ord("N"))
_MISSING_CODE = np.uint8(ord("."))
_IGNORED_CODES = (_N_CODE, _MISSING_CODE)

# IUPAC ambiguity codes as byte values, keyed by the sorted constituent bases.
_IUPAC_CODE_BYTES = {k: np.uint8(ord(v)) for k, v in IUPAC_CODES.items()}

_QUERY_COLUMNS = ["chrom", "pos", "ref", "alt", "dp", "gt"]


def open_fasta(file: File, mode: str = "r"):
    """
    Open a FASTA file, transparently handling gzip compression.

    Parameters
    ----------
    file : File
        Path to the FASTA file. Treated as gzip-compressed if it ends in ".gz".
    mode : str, optional
        Mode to open the file in. Default is "r".
    """
    if str(file).endswith(".gz"):
        return gzip.open(file, mode=mode + "t")

    return open(file, mode=mode)


def load_reference(
    reference: File
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int]]:
    """
    Read a reference FASTA into a flat array of base codes.

    Contigs are concatenated in the order they appear in the file, so a VCF
    record maps to the array index ``offsets[chrom] + pos - 1``.

    Parameters
    ----------
    reference : File
        Path to the reference FASTA, optionally gzip-compressed.

    Returns
    -------
    tuple
        The concatenated sequence as an array of base codes, a mapping of contig
        name to its start index in that array, and a mapping of contig name to
        its length.
    """
    chunks, offsets, lengths, cursor = [], {}, {}, 0

    with open_fasta(reference) as handle:
        for record in SeqIO.parse(handle, "fasta"):

            if record.id in offsets:
                raise ValueError(
                    f"Reference FASTA {str(reference)} has more than one contig named "
                    f"{record.id}."
                )

            sequence = np.frombuffer(
                str(record.seq).upper().encode("ascii"), dtype=np.uint8
            )

            offsets[record.id] = cursor
            lengths[record.id] = len(sequence)
            cursor += len(sequence)
            chunks.append(sequence)

    if not chunks:
        raise ValueError(f"Reference FASTA {str(reference)} has no sequences.")

    return np.concatenate(chunks), offsets, lengths


def include_expression(min_dp: int) -> str:
    """
    Build the bcftools expression selecting the records that need to be read.

    Any record that could resolve to something other than the reference base is
    matched: alternate and missing genotypes, sites below the coverage
    threshold, sites with no depth annotation, indels, and multi-allelic sites.
    Records that do not match are single-base reference calls with adequate
    coverage, and the alignment already holds the reference base for those.

    The expression is deliberately permissive - it selects a superset and
    `resolve_bases` does the authoritative classification - because a record
    wrongly skipped here would silently become a reference base.

    Parameters
    ----------
    min_dp : int
        Minimum depth of coverage for a genotype call to be trusted.
    """
    return (
        f'GT="alt" || GT="mis" || INFO/DP<{int(min_dp)} || INFO/DP="." '
        '|| strlen(REF)!=1 || strlen(ALT)!=1 || N_ALT>1'
    )


def vcf_samples(vcf: File) -> List[str]:
    """
    Return the sample names declared in the header of a VCF.

    Parameters
    ----------
    vcf : File
        Path to the VCF file, optionally compressed.
    """
    header = VCF(vcf).get_header()
    for line in header.splitlines():
        if line.startswith("#CHROM"):
            return line.split("\t")[9:]
    return []


def _normalize_vcf_df(vcf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map ``VCF.read()`` output to the ``chrom/pos/ref/alt/dp/gt`` format.

    The sample column is always the last column in the DataFrame; its name
    depends on what the VCF header declares.  The GT value is the first
    colon-separated field of the sample cell (in case FORMAT has more fields).
    The DP value is extracted from the INFO string.
    """
    sample_col = vcf_df.columns[-1]
    gt = vcf_df[sample_col].astype(str).str.split(":").str[0]
    dp = vcf_df["info"].apply(lambda x: get_info_value(x, "DP", dtype=str))
    return pd.DataFrame({
        "chrom": vcf_df["chrom"].values,
        "pos": vcf_df["pos"].values,
        "ref": vcf_df["ref"].values,
        "alt": vcf_df["alt"].values,
        "dp": dp.values,
        "gt": gt.values,
    })

def read_sparse_vcf(vcf: File, min_dp: int = 10) -> pd.DataFrame:
    """
    Read the records of a VCF that do not resolve to the reference base.

    Filtering happens inside bcftools, so only the selected records are ever
    decoded into Python.

    Parameters
    ----------
    vcf : File
        Path to the VCF file, optionally compressed.
    min_dp : int, optional
        Minimum depth of coverage for a genotype call to be trusted.
        Default is 10.

    Returns
    -------
    pd.DataFrame
        One row per selected record, with columns chrom, pos, ref, alt, dp and
        gt. Empty if no record was selected.
    """
    samples = vcf_samples(vcf)

    if len(samples) != 1:
        raise ValueError(
            f"Expected exactly one sample per VCF, but {str(vcf)} has {len(samples)}"
            + (f" ({', '.join(samples)})" if samples else "")
            + ". CleanSweep collection takes one single-sample VCF per plate swipe."
        )

    vcf_df = VCF(vcf).read(
        chrom=None,
        include=[include_expression(min_dp)],
        add_base_counts=False,
        filter_indels=False,
    )

    if vcf_df.empty:
        return pd.DataFrame(columns=_QUERY_COLUMNS)

    return _normalize_vcf_df(vcf_df)


def read_reference_calls(
    vcf: File,
    min_dp: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the records of a VCF that do resolve to the reference base.

    This is the complement of `read_sparse_vcf`, and is only needed on the
    reference-free path, where the reference bases have to come from the REF
    column of the VCF itself.

    A record on which the expression cannot be evaluated at all - one with no
    FORMAT/GT field, say - satisfies neither the include nor the exclude form,
    so it appears in neither result and is left as "N". That matches how the
    row-wise implementation treated records it could not classify.

    Parameters
    ----------
    vcf : File
        Path to the VCF file, optionally compressed.
    min_dp : int, optional
        Minimum depth of coverage for a genotype call to be trusted.
        Default is 10.

    Returns
    -------
    tuple
        Positions of the reference calls and their base codes.
    """
    vcf_df = VCF(vcf).read(
        chrom=None,
        exclude=[include_expression(min_dp)],
        add_base_counts=False,
        filter_indels=False,
    )

    if vcf_df.empty:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint8)

    return vcf_df["pos"].to_numpy(dtype=np.int64), first_base_codes(vcf_df["ref"])


def first_base_codes(values: pd.Series) -> np.ndarray:
    """
    Return the base code of the first character of every string in a Series.

    Parameters
    ----------
    values : pd.Series
        Series of allele strings.
    """
    # The trailing fillna catches empty strings, which str[0] turns into NaN.
    first = values.fillna("N").astype(str).str.upper().str[0].fillna("N")

    return np.frombuffer(first.to_numpy(dtype="S1").tobytes(), dtype=np.uint8)


def resolve_bases(vcf_df: pd.DataFrame, min_dp: int = 10) -> np.ndarray:
    """
    Map each VCF record to the single base it contributes to the alignment.

    Precedence, matching the row-wise implementation this replaces:

    - depth below `min_dp`, or no depth annotation, becomes "N"
    - a missing genotype becomes "N"
    - an indel or multi-allelic site becomes "N"
    - genotype 1 becomes the alternate allele
    - genotype 0 becomes the reference allele
    - anything else becomes "N"

    Parameters
    ----------
    vcf_df : pd.DataFrame
        Records as returned by `read_sparse_vcf`.
    min_dp : int, optional
        Minimum depth of coverage for a genotype call to be trusted.
        Default is 10.

    Returns
    -------
    np.ndarray
        One base code per record.
    """
    reference_codes = first_base_codes(vcf_df["ref"])
    alternate_codes = first_base_codes(vcf_df["alt"])

    # A missing depth annotation coerces to NaN, which fails the comparison and
    # so counts as insufficient coverage.
    depth = pd.to_numeric(vcf_df["dp"], errors="coerce").to_numpy(dtype=float)
    covered = depth >= min_dp

    # Single-base REF and ALT only: indels and multi-allelic sites cannot be
    # placed in a fixed-width alignment.
    is_snv = (
        vcf_df["ref"].fillna("").astype(str).str.len().eq(1)
        & vcf_df["alt"].fillna("").astype(str).str.len().eq(1)
    ).to_numpy()

    genotype = vcf_df["gt"].fillna(".").astype(str)
    usable = is_snv & covered

    return np.where(
        usable & genotype.eq("1").to_numpy(),
        alternate_codes,
        np.where(usable & genotype.eq("0").to_numpy(), reference_codes, _N_CODE),
    ).astype(np.uint8)


def combine_duplicates(
    positions: np.ndarray,
    codes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapse records sharing a position into a single base code.

    Positions carrying more than one distinct base are represented with the
    corresponding IUPAC ambiguity code. A position where any record is ambiguous
    stays ambiguous.

    Parameters
    ----------
    positions : np.ndarray
        Alignment index of each record.
    codes : np.ndarray
        Base code of each record.

    Returns
    -------
    tuple
        Sorted unique positions and their combined base codes.
    """
    order = np.argsort(positions, kind="stable")
    positions, codes = positions[order], codes[order]

    unique, starts, counts = np.unique(
        positions, return_index=True, return_counts=True
    )

    combined = codes[starts].copy()

    # Only positions carrying more than one record need combining, which is rare.
    for j in np.flatnonzero(counts > 1):
        group = codes[starts[j]:starts[j] + counts[j]]

        if np.isin(group, _IGNORED_CODES).any():
            combined[j] = _N_CODE
            continue

        bases = "".join(sorted({chr(code) for code in group}))
        combined[j] = _IUPAC_CODE_BYTES.get(bases, _N_CODE)

    return unique, combined


def pack_sequences(sequences: Iterable[str]) -> Tuple[np.ndarray, int]:
    """
    Pack sequences into an array of base codes, padding with "N".

    Parameters
    ----------
    sequences : Iterable[str]
        Nucleotide sequences, which need not be the same length.

    Returns
    -------
    tuple
        A ``sequences x length`` array of base codes and the alignment length.
    """
    sequences = list(sequences)

    if not sequences:
        return np.empty((0, 0), dtype=np.uint8), 0

    length = max(len(sequence) for sequence in sequences)
    matrix = np.full((len(sequences), length), _N_CODE, dtype=np.uint8)

    for i, sequence in enumerate(sequences):
        if sequence:
            matrix[i, :len(sequence)] = np.frombuffer(
                sequence.encode("ascii"), dtype=np.uint8
            )

    return matrix, length


def sequences_to_matrix(
    sequences: Dict[str, str]
) -> Tuple[List[str], np.ndarray, int]:
    """
    Pack a mapping of sequences into an array of base codes.

    Parameters
    ----------
    sequences : dict[str, str]
        A mapping of sequence name to nucleotide sequence.

    Returns
    -------
    tuple
        The sequence names in insertion order, a ``names x length`` array of
        base codes, and the alignment length.
    """
    names = list(sequences)
    matrix, length = pack_sequences(sequences[name] for name in names)

    return names, matrix, length


@dataclass
class Collection:

    vcfs: List[File]
    output: File
    reference: Union[File, None] = None
    alpha: float = 10.0
    min_coverage: int = 10
    exclude: bool = False
    exclude_log: Union[File, None] = None
    n_threads: Union[None, int] = None

    def __post_init__(self):

        # Check if the VCFs exist
        for vcf in self.vcfs:

            if not Path(vcf).exists():
                raise FileNotFoundError(f"VCF {str(vcf)} not found.")

        if self.reference is not None and not Path(self.reference).exists():
            raise FileNotFoundError(f"Reference FASTA {str(self.reference)} not found.")

        if self.alpha <= 0:
            raise ValueError(f"Alpha must be greater than 0. Got {self.alpha}.")

        if self.exclude_log is not None and not self.exclude:
            raise ValueError(f"--exclude-log ({self.exclude_log}) was given without --exclude.")

        self.__set_n_threads()

    def msa(self):
        """
        Converts a set of VCF files to a multiple sequence alignment (MSA) in FASTA
        format. Samples that are identified as outliers based on their pairwise
        similarities will be excluded from the MSA or have their private SNPs removed,
        depending on the `exclude` parameter.

        If `exclude` is False (default), this method looks for outlier samples: it
        calculates the maximum average nucleotide identity (ANI) each sample shares with
        any other sample. If a sample's maximum ANI is below the threshold defined by
        the median minus `alpha` times the interquartile range (IQR) of the maximum ANI
        values, it is considered an outlier. For each outlier sample, any SNPs that are
        not shared with at least one other sample (i.e., private SNPs) are removed.

        If `exclude` is True, outlier samples are completely excluded from the MSA.

        If a reference FASTA was given, the alignment spans the whole reference and only
        variant and low-coverage sites are held in memory. Otherwise each VCF is
        converted to a full-length sequence, which is considerably slower and uses far
        more memory for large collections.
        """
        if self.reference is None:
            logging.info(
                "No reference FASTA given, so each VCF will be converted to a "
                "full-length sequence. Pass a reference for a much faster run on "
                "large collections."
            )
            names, matrix, positions, reference = self.__build_dense()
        else:
            names, matrix, positions, reference = self.__build_sparse()

        if not names:
            warnings.warn("No sequences to align. Writing an empty MSA.")
            Path(self.output).write_text("")
            return

        total_length = len(reference) if reference is not None else matrix.shape[1]

        # Find outliers
        outlier_indices = self.outlier_indices(
            matrix=matrix,
            total_length=total_length,
            alpha=self.alpha,
            n_threads=self.n_threads
        )
        outliers = [names[i] for i in outlier_indices]

        if self.exclude and outliers:
            logging.info(
                f"Excluding {len(outliers)} outlier sample(s) from MSA: "
                f"{', '.join(outliers)}."
            )

            keep = np.setdiff1d(np.arange(len(names)), outlier_indices)
            names = [names[i] for i in keep]
            matrix = matrix[keep]

            # Write a list of excluded samples to the exclude log file if specified
            if self.exclude_log is not None:
                with open(self.exclude_log, "w") as f:
                    for name in outliers:
                        f.write(f"{name}\n")

        elif outliers:
            logging.info(
                f"Found {len(outliers)} outlier sample(s) in MSA: "
                f"{', '.join(outliers)}. Removing sample-private SNPs..."
            )

            if len(names) < 3:
                warnings.warn(
                    "You are trying to use CleanSweep collection with less than 3 "
                    "sequences. This may result in unreliable SNP calls."
                )

            # Generate consensus sequence
            consensus = self.consensus_from_matrix(matrix)

            # Remove private SNPs from every outlier, against the same consensus
            for i in outlier_indices:
                self.remove_private_from_matrix(
                    matrix=matrix,
                    row=i,
                    consensus=consensus
                )

        # Write MSA to output
        self.write_matrix(
            names=names,
            matrix=matrix,
            positions=positions,
            reference=reference
        )

    def __build_sparse(self):
        """
        Read every VCF into an alignment anchored on the reference.

        Only positions that differ from the reference in at least one sample are
        held, so the alignment array is proportional to the number of variant and
        low-coverage sites rather than to the genome length.
        """
        reference, offsets, lengths = load_reference(self.reference)

        with ThreadPool(processes=self.n_threads) as pool:
            parsed = pool.starmap(
                self._vcf_to_sparse_item,
                [(vcf, offsets, lengths) for vcf in self.vcfs]
            )

        names = [name for name, _, _ in parsed]
        indices = [index for _, index, _ in parsed]

        populated = [index for index in indices if index.size]
        positions = (
            np.unique(np.concatenate(populated))
            if populated
            else np.empty(0, dtype=np.int64)
        )

        logging.info(
            f"Found {positions.size} variant or low-coverage site(s) across "
            f"{len(names)} sample(s) in a {len(reference)} bp reference."
        )

        # Seeding with the reference is what lets the untouched positions be left
        # out: every sample holds the reference base there by definition.
        matrix = np.tile(reference[positions], (len(names), 1))

        for i, (_, index, codes) in enumerate(parsed):
            if index.size:
                matrix[i, np.searchsorted(positions, index)] = codes

        return names, matrix, positions, reference

    def __build_dense(self):
        """Convert every VCF to a full-length sequence, without a reference."""
        with ThreadPool(processes=self.n_threads) as pool:
            sequences = dict(pool.starmap(
                self._vcf_to_seq_item, [(vcf,) for vcf in self.vcfs]
            ))

        names, matrix, _ = sequences_to_matrix(sequences)

        return names, matrix, None, None

    def _vcf_to_sparse_item(self, vcf, offsets, lengths):
        return (Path(vcf).stem,) + self.vcf_to_sparse(
            vcf=vcf,
            offsets=offsets,
            lengths=lengths,
            min_dp=self.min_coverage
        )

    def _vcf_to_seq_item(self, vcf):
        return (Path(vcf).stem, self.vcf_to_seq(vcf=vcf, min_dp=self.min_coverage))

    def vcf_to_sparse(
        self,
        vcf: File,
        offsets: Dict[str, int],
        lengths: Dict[str, int],
        min_dp: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a VCF to the positions and bases that differ from the reference.

        Parameters
        ----------
        vcf : File
            Path to the input VCF file.
        offsets : dict[str, int]
            Mapping of contig name to its start index in the reference array, as
            returned by `load_reference`.
        lengths : dict[str, int]
            Mapping of contig name to its length, as returned by
            `load_reference`.
        min_dp : int, optional
            Minimum depth of coverage to consider a genotype call valid.
            Default is 10.

        Returns
        -------
        tuple
            Sorted alignment indices and their base codes.
        """
        vcf_df = read_sparse_vcf(vcf, min_dp=min_dp)

        if vcf_df.empty:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint8)

        unknown = set(vcf_df["chrom"].unique()) - set(offsets)
        if unknown:
            raise ValueError(
                f"VCF {str(vcf)} has records on contig(s) absent from the reference: "
                f"{', '.join(sorted(unknown))}. The reference has "
                f"{', '.join(sorted(offsets))}."
            )

        positions = vcf_df["pos"].to_numpy(dtype=np.int64)
        contig_lengths = vcf_df["chrom"].map(lengths).to_numpy(dtype=np.int64)

        outside = (positions < 1) | (positions > contig_lengths)
        if outside.any():
            record = vcf_df[outside].iloc[0]
            raise ValueError(
                f"VCF {str(vcf)} has a record at {record.chrom}:{record.pos}, outside "
                f"the reference contig (1-{lengths[record.chrom]})."
            )

        indices = vcf_df["chrom"].map(offsets).to_numpy(dtype=np.int64) + positions - 1

        return combine_duplicates(indices, resolve_bases(vcf_df, min_dp=min_dp))

    def vcf_to_seq(
        self,
        vcf: File,
        min_dp: int = 10,
    ) -> str:
        """
        Convert a VCF file to a nucleotide sequence. Handles multi-allelic sites by
        using IUPAC codes to represent the bases.

        This is the reference-free path: the sequence runs from position 1 to the last
        position in the VCF, and positions with no record at all are left as "N".
        Genotypes are taken from the FORMAT/GT field. Prefer `vcf_to_sparse` when a
        reference FASTA is available.

        Parameters
        ----------
        vcf : File
            Path to the input VCF file.
        min_dp : int, optional
            Minimum depth of coverage to consider a genotype call valid. Default is 10.

        Returns
        -------
        str
            A string representing the nucleotide sequence, where each position
            corresponds to a base in the reference genome.
        """
        variants = read_sparse_vcf(vcf, min_dp=min_dp)
        reference_positions, reference_codes = read_reference_calls(vcf, min_dp=min_dp)

        variant_positions, variant_codes = (
            combine_duplicates(
                variants["pos"].to_numpy(dtype=np.int64),
                resolve_bases(variants, min_dp=min_dp)
            )
            if not variants.empty
            else (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint8))
        )

        # The two queries partition the VCF, so between them they see every record.
        last_position = max(
            int(variant_positions.max()) if variant_positions.size else 0,
            int(reference_positions.max()) if reference_positions.size else 0,
        )

        # Return an empty string if the VCF is empty
        if last_position == 0:
            return ""

        sequence = np.full(last_position, _N_CODE, dtype=np.uint8)
        sequence[reference_positions - 1] = reference_codes
        sequence[variant_positions - 1] = variant_codes

        return sequence.tobytes().decode("ascii")

    @staticmethod
    def max_identities(
        matrix: np.ndarray,
        total_length: int,
        n_threads: int = 1
    ) -> np.ndarray:
        """
        Highest fraction of identical positions each row shares with any other row.

        Only the columns present in `matrix` are compared, but the fraction is taken
        over `total_length`. Any column left out of a reference-anchored alignment
        holds the reference base in every sample and so contributes no differences,
        which makes this exactly equal to a whole-genome Hamming identity.

        Parameters
        ----------
        matrix : np.ndarray
            A ``samples x positions`` array of base codes.
        total_length : int
            Length of the full alignment, used as the denominator.
        n_threads : int, optional
            Number of threads to compare rows with. Default is 1.

        Returns
        -------
        np.ndarray
            One maximum identity per row.
        """
        n_samples = matrix.shape[0]

        if n_samples < 2 or total_length <= 0:
            return np.ones(n_samples)

        def min_differences(i: int) -> int:
            differences = np.count_nonzero(matrix != matrix[i], axis=1)
            # A row is trivially identical to itself, so rule it out.
            differences[i] = np.iinfo(differences.dtype).max
            return int(differences.min())

        with ThreadPool(processes=n_threads) as pool:
            minima = pool.map(min_differences, range(n_samples))

        return 1.0 - np.asarray(minima, dtype=float) / float(total_length)

    @classmethod
    def outlier_indices(
        cls,
        matrix: np.ndarray,
        total_length: int,
        alpha: float = 3.0,
        n_threads: int = 1
    ) -> List[int]:
        """
        Identify rows whose maximum identity to any other row is an outlier.

        Parameters
        ----------
        matrix : np.ndarray
            A ``samples x positions`` array of base codes.
        total_length : int
            Length of the full alignment.
        alpha : float, optional
            The multiplier for the interquartile range (IQR) to determine outliers.
            Default is 3.0.
        n_threads : int, optional
            Number of threads to compare rows with. Default is 1.

        Returns
        -------
        list[int]
            Row indices identified as outliers.
        """
        if matrix.shape[0] < 2 or total_length <= 0:
            return []

        identities = cls.max_identities(
            matrix=matrix,
            total_length=total_length,
            n_threads=n_threads
        )

        # Identify outliers based on the IQR method
        median = np.median(identities)
        q1, q3 = np.percentile(identities, [25, 75])

        return np.flatnonzero(identities < (median - alpha * (q3 - q1))).tolist()

    @staticmethod
    def consensus_from_matrix(matrix: np.ndarray) -> np.ndarray:
        """
        Per-column most common base code, ignoring "N" and ".".

        Parameters
        ----------
        matrix : np.ndarray
            A ``samples x positions`` array of base codes.

        Returns
        -------
        np.ndarray
            One consensus base code per column. Columns with no unambiguous base
            at all are "N".
        """
        if matrix.shape[0] == 0:
            raise ValueError("No sequences provided for consensus generation.")

        consensus = np.full(matrix.shape[1], _N_CODE, dtype=np.uint8)
        best = np.zeros(matrix.shape[1], dtype=np.int64)

        # Ascending order plus a strict comparison below means a tie resolves to
        # the lexicographically smallest base.
        for code in sorted(
            code for code in np.unique(matrix) if code not in _IGNORED_CODES
        ):
            counts = np.count_nonzero(matrix == code, axis=0)

            better = counts > best
            consensus[better] = code
            best[better] = counts[better]

        return consensus

    @staticmethod
    def remove_private_from_matrix(
        matrix: np.ndarray,
        row: int,
        consensus: np.ndarray
    ):
        """
        Replace the bases unique to one row with the consensus, in place.

        A row compares equal to itself, so requiring every other row to differ is
        the same as comparing against the matrix with that row dropped - without
        having to copy it out.

        Parameters
        ----------
        matrix : np.ndarray
            A ``samples x positions`` array of base codes, modified in place.
        row : int
            Index of the row to clean.
        consensus : np.ndarray
            Consensus base codes, as returned by `consensus_from_matrix`.
        """
        n_samples = matrix.shape[0]

        if n_samples < 2:
            return

        private = np.count_nonzero(matrix != matrix[row], axis=0) == n_samples - 1
        matrix[row, private] = consensus[private]

    def write_matrix(
        self,
        names: List[str],
        matrix: np.ndarray,
        positions: Union[np.ndarray, None],
        reference: Union[np.ndarray, None]
    ):
        """
        Write an alignment held as an array of base codes to a FASTA file.

        With a reference, each sequence is the reference with the alignment columns
        patched into it, so only one full-length sequence is ever resident.

        Parameters
        ----------
        names : list[str]
            Sequence names, in the same order as the rows of `matrix`.
        matrix : np.ndarray
            A ``samples x positions`` array of base codes.
        positions : np.ndarray or None
            Indices of the alignment columns within the reference. None when
            `matrix` already holds full-length sequences.
        reference : np.ndarray or None
            Reference base codes, or None for a reference-free alignment.
        """
        with open(self.output, "wb") as f:
            for i, name in enumerate(names):

                if reference is None:
                    sequence = matrix[i]
                else:
                    sequence = reference.copy()
                    sequence[positions] = matrix[i]

                f.write(b">" + str(name).encode("ascii") + b"\n")
                f.write(sequence.tobytes())
                f.write(b"\n")

    def __set_n_threads(self):
        if self.n_threads is None:
            self.n_threads = cpu_count()
        elif self.n_threads < 1:
            raise ValueError(f"n_threads must be greater than 0. Got {self.n_threads}.")
        elif self.n_threads > cpu_count():
            warnings.warn(
                f"n_threads ({self.n_threads}) is greater than the number of available "
                f"CPU cores ({cpu_count()}). Using {cpu_count()} threads instead."
            )
            self.n_threads = cpu_count()


# %%
