from datetime import date
from functools import partial
import sys
import pandas as pd
import subprocess
from typing import List, Union, Collection
import numpy as np 
from dataclasses import dataclass
from pathlib import Path
import logging
from io import StringIO
from warnings import warn, catch_warnings, simplefilter
from cleansweep.typing import File
from cleansweep.__version__ import __version__

_VCF_HEADER = ["chrom", "pos", "id", "ref", "alt", "qual", "filter", "info", "format", "sample"]

# Order of the bases in the Pilon BC (base counts) INFO tag
_BASES = ["A", "C", "G", "T"]
_BASE_INDEX = {b: i for i, b in enumerate(_BASES)}
_BC_COLS = ["bc_A", "bc_C", "bc_G", "bc_T"]


@dataclass
class VCF:

    file: File

    def __post_init__(self):

        if isinstance(self.file, Path): self.file = str(self.file) 

    def read(
        self,
        chrom: Union[str, None, List],
        filters: Union[Collection[str], None] = None,
        include: Union[str, None]=None,
        exclude: Union[str, None]=None,
        add_base_counts: bool = True,
        collapse: bool = True
    ) -> pd.DataFrame:
        """Filters a VCF file (compressed or uncompressed) with bcftools. Requires bcftools.

        Args:
            filters (Union[Collection[str], None], optional): Records with these values in the \
FILTER field are included; all others are excluded. If None, includes all values. Defaults to None.
            include (Union[str, None], optional): bcftools expression defining records to include. If \
None, no filters are applied. Defaults to None.
            exclude (Union[str, None], optional): bcftools expression defining records to exclude. If \
None, no filters are applied. Defaults to None.
        """

        filters_cmd = ["-f"] + list(filters) if filters is not None else []
        include_cmd = ["-i"] + list(include) if include is not None else []
        exclude_cmd = ["-e"] + list(exclude) if exclude is not None else []
        command = ["bcftools", "view"] + filters_cmd + include_cmd + exclude_cmd + ["-O", "v", self.file]

        # Run command
        logging.info(f"Running \"{' '.join(command)}\"...")
        rc = subprocess.run(command, stdout=subprocess.PIPE)

        if rc.returncode != 0:
            msg = f"Filtering the VCF file in {self.file} with bcftools failed. Got return \
code {rc.returncode}. Command: \'{' '.join(command)}\'."
            logging.error(msg)
            logging.error("stout dump:")
            logging.error(rc.stdout)
            raise RuntimeError(msg)
        
        # Store filtered VCF as an attribute
        self.vcf = pd.read_csv(
            StringIO(rc.stdout.decode("utf-8")), 
            sep="\t", 
            comment="#", 
            header=None
        )  

        self.vcf.columns = [
            x
            for x in rc.stdout \
                .decode("utf-8") \
                .split("\n") 
            if x.startswith("#")
        ][-1].split("\t")

        self.vcf.columns = [
            (
                x.removeprefix("#").lower()
                if x.removeprefix("#").lower() in _VCF_HEADER
                else x.removeprefix("#")
            ) for x in self.vcf.columns
        ]

        # Keep variants in the query
        if chrom:
            if isinstance(chrom, str): chrom = [chrom]
            self.vcf = self.vcf[self.vcf.chrom.isin(chrom)]

        self.vcf = self.exclude_indels(self.vcf)
        if add_base_counts:
            self.vcf = self.add_info_columns(self.vcf)
            # Collapse multiallelic sites split across several caller lines into one
            # row per site (keeps the per-allele base counts on a single record).
            # CleanSweep output VCFs intentionally split multiallelic sites across
            # lines, so reading those back should pass collapse=False.
            if collapse:
                self.vcf = self.collapse_multiallelic_lines(self.vcf)

        return self.vcf
    
    def get_header(
        self
    ) -> str:
        """Returns the header of this VCF.

        Returns:
            str: Header
        """
        
        command = ["bcftools", "view", str(self.file), "-h"]

        # Run command
        logging.info(f"Running \"{' '.join(command)}\"...")
        rc = subprocess.run(command, stdout=subprocess.PIPE)

        if rc.returncode != 0:
            msg = f"Filtering the VCF file in {self.file} with bcftools failed. Got return \
code {rc.returncode}. Command: \'{' '.join(command)}\'."
            logging.error(msg)
            logging.error("stout dump:")
            logging.error(rc.stdout)
            raise RuntimeError(msg)
        
        return rc.stdout.decode("utf-8")

    def exclude_indels(self, vcf: Union[None, pd.DataFrame]=None) -> pd.DataFrame:

        if vcf is None: 
            inplace = True
            vcf = self.vcf
        else:
            inplace = False

        vcf = vcf[vcf.ref.str.len().eq(1) & vcf.alt.str.len().eq(1)]
        if inplace: self.vcf = vcf
        return vcf
        
    def include_chrom(self, chroms: Collection[str], 
                      vcf: Union[None, pd.DataFrame]=None) -> pd.DataFrame:

        if vcf is None: 
            inplace = True
            vcf = self.vcf

        vcf = vcf[vcf.chrom.isin(chroms)]
        if inplace: self.vcf = vcf
        return vcf
    
    def add_info_columns(self, vcf: pd.DataFrame) -> pd.DataFrame:
        """Adds the number of bases supporting the reference and alternate allele and the mean mapping
        quality per site as columns to a VCF DataFrame.

        Args:
            vcf (pd.DataFrame): VCF Pandas DataFrame.

        Returns:
            pd.DataFrame: VCF Pandas DataFrame with reference and allele basecounts (
                columns \"ref_bc\" and \"alt_bc\", respectively), and mapping quality (\"mapq\").
        """

        if not hasattr(vcf, "base_counts"):
            vcf = vcf.assign(base_counts = vcf["info"] \
                .apply(partial(get_info_value, tag="BC", dtype=str)))

        # Parse the per-allele base counts (bc_A, bc_C, bc_G, bc_T) for
        # multiallelic site detection
        vcf = self.parse_base_counts(vcf)

        if not hasattr(vcf, "mapq"):
            vcf = vcf.assign(mapq = vcf["info"] \
                .apply(partial(get_info_value, tag="MQ", dtype=int)))
            
        if not hasattr(vcf, "depth"):
            vcf = vcf.assign(depth = vcf["info"] \
                .apply(partial(get_info_value, tag="DP", dtype=int)))
            
        if not hasattr(vcf, "p_alt"):
            vcf = vcf.assign(p_alt = vcf["info"] \
                .apply(
                    lambda x: (
                        get_info_value(x, tag="CSP", dtype=int)
                        if "CSP=" in x and x != "."
                        else pd.NA
                    )
                )
            )

        # Parse the multiallelic INFO fields written by the CleanSweep filter so that
        # downstream tools (e.g. inspect) can read them back from the output VCF
        for tag, column in [
            ("AF", "allele_fraction"),
            ("PMULTI", "p_multi"),
            ("LLR", "llr"),
            ("LL", "best_logL"),
        ]:
            if column not in vcf.columns:
                vcf = vcf.assign(
                    **{
                        column: vcf["info"].apply(
                            partial(get_info_value, tag=tag, dtype=float)
                        )
                    }
                )

        if "is_multiallelic" not in vcf.columns:
            vcf = vcf.assign(
                is_multiallelic = vcf["info"].apply(
                    lambda x: bool(_has_info_flag(x, "MULTI"))
                )
            )

        # If a variant is missing alternate allele information, extract is from the base counts
        vcf.loc[
            vcf.alt.eq("."),
            "alt"
        ] = vcf[
            vcf.alt.eq(".")
        ].apply(
            lambda x: self.__alt_from_base_counts(
                x.base_counts,
                x.ref
            ),
            axis = 1
        )
    
        return vcf.assign(
            alt_bc = vcf.apply(
                lambda x: self.extract_base_counts(
                    x.base_counts,
                    x.alt
                ),
                axis = 1
            ),
            ref_bc = vcf.apply(
                lambda x: self.extract_base_counts(
                    x.base_counts,
                    x.ref
                ),
                axis = 1
            )
        )

    def collapse_multiallelic_lines(self, vcf: pd.DataFrame) -> pd.DataFrame:
        """Collapses multiallelic sites split across several lines into a single row.

        Some variant callers emit one line per alternate allele, so a multiallelic site
        can appear as multiple records sharing the same CHROM and POS. CleanSweep needs
        one row per site holding the per-allele base counts. Pilon already reports a
        site-level BC tag, so the base counts are identical across split lines; taking
        the maximum across lines is robust both to that case and to callers that report
        per-line counts.

        Args:
            vcf (pd.DataFrame): VCF DataFrame with the per-allele ``bc_*`` columns.

        Returns:
            pd.DataFrame: One row per (chrom, pos), with an ``alt_set`` column listing the
                distinct alternate alleles seen across the collapsed lines.
        """

        # Record the distinct alternate alleles seen at each site
        vcf = vcf.assign(
            alt_set = vcf["alt"].apply(
                lambda a: frozenset([a]) 
                if isinstance(a, str) and a != "." 
                else frozenset()
            )
        )

        # Nothing to collapse if there is at most one line per site
        site_id = vcf["chrom"].astype(str) + "_" + vcf["pos"].astype(str)
        if not site_id.duplicated().any():
            return vcf

        logging.debug(
            f"Collapsing {int(site_id.duplicated().sum())} multiallelic lines split "
            "across records."
        )

        aggregations = {
            col: ("max" if col in _BC_COLS else "first")
            for col in vcf.columns
            if col not in ["chrom", "pos", "alt_set"]
        }
        aggregations["alt_set"] = lambda s: frozenset().union(*s)

        collapsed = vcf \
            .groupby(["chrom", "pos"], sort=False, as_index=False) \
            .agg(aggregations)

        # Recompute the per-allele base counts (alt/ref) after collapsing, using the
        # representative alt allele kept from the first line
        return collapsed.assign(
            alt_bc = collapsed.apply(
                lambda x: self.extract_base_counts(x.base_counts, x.alt),
                axis = 1
            ),
            ref_bc = collapsed.apply(
                lambda x: self.extract_base_counts(x.base_counts, x.ref),
                axis = 1
            )
        )

    def parse_base_counts(self, vcf: pd.DataFrame) -> pd.DataFrame:
        """Splits the BC base counts string (\"A,C,G,T\") into four integer columns.

        Adds nullable integer columns ``bc_A``, ``bc_C``, ``bc_G`` and ``bc_T`` so the
        per-allele read counts are available for multiallelic site detection. Sites with
        a missing or malformed BC tag get ``<NA>`` for every base count.

        Args:
            vcf (pd.DataFrame): VCF DataFrame with a ``base_counts`` column.

        Returns:
            pd.DataFrame: The same DataFrame with the four ``bc_*`` columns added.
        """

        if all(c in vcf.columns for c in _BC_COLS):
            return vcf

        bc = vcf["base_counts"] \
            .str.split(",", expand=True)

        # Guard against malformed BC tags that do not have all four bases
        for i, col in enumerate(_BC_COLS):
            vcf = vcf.assign(
                **{
                    col: (
                        pd.to_numeric(bc[i], errors="coerce").astype("Int64")
                        if i in bc.columns
                        else pd.array([pd.NA] * len(vcf), dtype="Int64")
                    )
                }
            )

        return vcf

    def extract_base_counts(
        self,
        base_counts: str,
        nucleotide: str,
    ) -> int:

        if isinstance(nucleotide, str) and nucleotide in _BASE_INDEX:
            return int(base_counts.split(",")[_BASE_INDEX[nucleotide]])
        else:
            warn(f"Got unknown base {nucleotide}. Setting base count to 0.")
            return 0
    
    def remove_indels(self, vcf: pd.DataFrame) -> pd.DataFrame:

        return vcf[vcf.alt.ne(".") & vcf.ref.ne(".")]
    
    def __alt_from_base_counts(
        self,
        base_counts: str,
        ref: str
    ) -> str:

        # Convert base counts to int, setting the base count for the ref allele
        # to the lowest count
        bc = [
            int(x) if b != ref else -1
            for x, b in zip(
                base_counts.split(","),
                _BASES
            )
        ]

        # Find the maximum base count (alt allele)
        return _BASES[np.argmax(bc)]
        
def get_info_value(s:str, tag:str, delim:str = ";", dtype = float):
    
    if not tag in s: return None

    str_value = s.split(tag+"=")[-1].split(delim)[0]

    try:
        return dtype(str_value)
    except:
        return None

def _has_info_flag(s: str, flag: str, delim: str = ";") -> bool:
    """Returns True if a valueless INFO flag (e.g. \"MULTI\") is present in an INFO string."""

    if not isinstance(s, str):
        return False

    return flag in [field.split("=")[0] for field in s.split(delim)]

def format_vcf_header(
    header: str,
    chrom: Union[None, str, List] = None,
    ref: Union[None, File, str] = None,
    add_filters: bool = True
) -> str:
    
    lines = []

    for line in header.split("\n"):

        if line.startswith("##fileformat="):
            # Update VCF version
            line = "##fileformat=VCFv4.2"

        elif line.startswith("##fileDate="):
            # Update date
            line = f"##fileDate={date.today().strftime('%Y%m%d')}"

        elif line.startswith("##source="):
            # Update source
            line = f"##source=\"CleanSweep version {__version__}\""

        lines.append(line)

    # Add descriptors for CleanSweep fields
    new_lines = [
        f"##CleanSweepCommand=\"{' '.join(sys.argv)}\""
    ] + (
        [
            "##FILTER=<ID=RefVar,Description=\"Variant also present in the references\">",
            "##FILTER=<ID=FAIL,Description=\"Alternate allele depth does not match the overall depth of coverage\">",
            "##FILTER=<ID=LowAltBC,Description=\"Alternate allele depth is too low\">",
            "##FILTER=<ID=LowCov,Description=\"Coverage is too low\">",
            "##INFO=<ID=BC,Number=4,Type=Integer,Description=\"Base counts\">",
            "##INFO=<ID=ORGFILT,Number=1,Type=String,Description=\"Original FILTER flag in the input VCF\">",
            "##INFO=<ID=CSP,Number=1,Type=Integer,Description=\"CleanSweep likelihood ratio for a variant being present in the query strain, log transformed\">",
            "##INFO=<ID=RD,Number=1,Type=Integer,Description=\"Reference allele base count\">",
            "##INFO=<ID=AD,Number=1,Type=Integer,Description=\"Alternate allele base count for this ALT allele\">",
            "##INFO=<ID=AF,Number=1,Type=Float,Description=\"CleanSweep estimated allele fraction of this ALT allele in the query population\">",
            "##INFO=<ID=LL,Number=1,Type=Float,Description=\"Log-likelihood of the most likely combination of alleles\">",
            "##INFO=<ID=LLR,Number=1,Type=Float,Description=\"Log-likelihood ratio between the most likely and second most likely combination of alleles\">",
            "##INFO=<ID=PMULTI,Number=1,Type=Float,Description=\"Probability that the site is multiallelic (microdiversity)\">",
            "##INFO=<ID=MULTI,Number=0,Type=Flag,Description=\"Site called multiallelic by CleanSweep\">",
        ]
        if add_filters
        else []
    )
        
    if not chrom is None:

        if isinstance(chrom, str): chrom = [chrom]
        new_lines.append(f"##CleanSweepChrom=\"{','.join(chrom)}\"")

    if not ref is None:
        new_lines.append(
            f"##CleanSweepReference=\"{str(ref)}\""
        )

    lines = lines[:-2] + new_lines + lines[-2:-1]

    return "\n".join(lines)

def expand_multiallelic(vcf: pd.DataFrame) -> pd.DataFrame:
    """Expands passing sites into one row per called (non-reference) allele.

    For each passing site whose most likely combination of alleles contains one or more
    non-reference alleles, a row is emitted per non-reference allele, with that row's
    ``alt``, ``alt_bc`` and ``allele_fraction`` set to the values of the called allele.
    Multiallelic sites therefore appear as several lines, one alternate allele per line.
    Sites that were not evaluated by the multiallelic model (missing ``best_combination``,
    e.g. rows coming from the un-evaluated full VCF) pass through unchanged.

    Args:
        vcf (pd.DataFrame): VCF DataFrame, optionally carrying the multiallelic model
            columns (``best_combination``, ``allele_fractions``, ``cleansweep_filter``).

    Returns:
        pd.DataFrame: VCF DataFrame with multiallelic sites expanded to one row per ALT.
    """

    if "best_combination" not in vcf.columns:
        return vcf

    def _is_expandable(row: pd.Series) -> bool:
        combination = row.get("best_combination")
        return (
            isinstance(combination, frozenset)
            and row.get("cleansweep_filter") == "PASS"
            and len(combination - {row["ref"]}) >= 1
        )

    expandable = vcf.apply(_is_expandable, axis=1)
    if not expandable.any():
        return vcf

    # Ensure the per-allele fraction column exists on every row so concatenating the
    # expanded rows back does not change column dtypes
    if "allele_fraction" not in vcf.columns:
        vcf = vcf.assign(allele_fraction = np.nan)

    kept = vcf[~expandable]
    rows = []
    for _, row in vcf[expandable].iterrows():
        fractions = row.get("allele_fractions") or {}
        base_counts = row.get("base_counts")
        for allele in sorted(row["best_combination"] - {row["ref"]}):
            new_row = row.copy()
            new_row["alt"] = allele
            if isinstance(base_counts, str):
                new_row["alt_bc"] = int(base_counts.split(",")[_BASE_INDEX[allele]])
            new_row["allele_fraction"] = (
                float(fractions[allele]) if allele in fractions else None
            )
            rows.append(new_row)

    expanded = pd.DataFrame(rows).reindex(columns=vcf.columns)

    # Some model columns are all-NA in the non-expanded rows; the resulting concat is
    # correct but pandas emits a dtype deprecation warning we can safely silence
    with catch_warnings():
        simplefilter("ignore", FutureWarning)
        return pd.concat([kept, expanded]).sort_values(["chrom", "pos"])

def _info_value(row: pd.Series, column: str, fmt) -> str:
    """Formats a per-site value for the INFO field, or \".\" when it is missing."""

    if column not in row:
        return "."

    value = row[column]
    if value is None or pd.isna(value):
        return "."

    return fmt(value)

def _build_info_field(row: pd.Series) -> str:
    """Builds the INFO column for an output VCF line, including multiallelic fields."""

    fields = [
        row["info"],
        "PILON=" + row["filter"],
        "CSP=" + _info_value(row, "p_alt", lambda v: str(int(v))),
        "RD=" + _info_value(row, "ref_bc", lambda v: str(int(v))),
        "AD=" + _info_value(row, "alt_bc", lambda v: str(int(v))),
        "AF=" + _info_value(row, "allele_fraction", lambda v: f"{v:.4f}"),
        "LL=" + _info_value(row, "best_logL", lambda v: f"{v:.4f}"),
        "LLR=" + _info_value(row, "llr", lambda v: f"{v:.4f}"),
        "PMULTI=" + _info_value(row, "p_multi", lambda v: f"{v:.4f}"),
    ]

    # MULTI is a valueless flag, only added to passing multiallelic sites
    if row.get("is_multiallelic") == True and row.get("cleansweep_filter") == "PASS":
        fields.append("MULTI")

    return ";".join(fields)

def write_vcf(
    vcf: pd.DataFrame,
    file: File,
    header: str,
    chrom: Union[None, str, List] = None,
):

    header = format_vcf_header(
        header,
        chrom = chrom
    )

    if not "cleansweep_filter" in vcf.columns:
        vcf = vcf.assign(
            cleansweep_filter = pd.NA
        )

    # Expand multiallelic sites into one row per called alternate allele
    vcf = expand_multiallelic(vcf)

    # Subset the columns in the VCF spec and add fields to INFO
    fmt_vcf = vcf[
        _VCF_HEADER
    ].rename(
        columns = {
            k: k.upper()
            for k in _VCF_HEADER
        }
    ).drop(
        columns = "FILTER"
    ).assign(
        FILTER = vcf["filter"].where(
                vcf.cleansweep_filter.isna(),
                vcf.cleansweep_filter
            ),
        SAMPLE = (
            vcf.cleansweep_filter \
                .eq("PASS") \
                .astype("Int8")
            if "cleansweep_filter" in vcf
            else vcf["sample"]
        ),
        INFO = vcf.apply(_build_info_field, axis = 1)
    )[
        [
            x.upper()
            for x in _VCF_HEADER
        ]
    ]

    # Convert DataFrame to TSV
    vcf_str = "\n".join(
        [
            "\t".join(x.astype(str)) 
            for x in fmt_vcf.values
        ]
    )

    # Write VCF
    with open(file, "w") as out:

        out.write(
            "\n".join(
                [
                    header,
                    vcf_str
                ]
            )
        )

def write_full_vcf(
    vcf: pd.DataFrame,
    full_vcf: File,
    file: File,
    header: str,
    *,
    chrom: Union[None, str] = None,
    min_dp: int = 0
):
    
    # Read full VCF
    full = VCF(full_vcf).read(
        chrom = chrom,
        add_base_counts = False
    )

    # Get the sample name
    sample_name = full.columns.to_list()[-1]

    # Fail sites with depth < min DP
    full = full.assign(
        filter = full["filter"].where(
            full["info"].apply(
                lambda x: get_info_value(x, "DP", dtype=int)
            ).ge(min_dp),
            "LowCov"
        )
    )

    # Pass ambiguous sites not evaluated by CleanSweep
    full.loc[
        full["filter"].str.contains("Amb"),
        "filter"
    ] = "PASS"

    # Set genotype info in sites not evaluated to ref
    full = full.assign(
        **{sample_name: "0"}
    )

    # Fail sites not meeting the alt depth threshold for CleanSweep evaluation
    full.loc[
        full["filter"].eq("PASS") & \
        (
            ~full.chrom.isin(vcf.chrom) | \
            ~full.pos.isin(vcf.pos)
        ),
        "filter"
    ] = "FAIL"

    # Exclude positions in the VCF to write
    in_out_vcf = (
        full.chrom + "_" + full.pos.astype(str)
    ).isin(
        vcf.chrom + "_" + vcf.pos.astype(str)
    )

    vcf = pd.concat(
        [
            full[~in_out_vcf],
            vcf
        ]
    ).sort_values(["chrom", "pos"])

    # Write to output
    write_vcf(
        vcf = vcf,
        file = file,
        header = header,
        chrom = chrom
    )

def write_merged_vcf(
    vcf: pd.DataFrame,
    file: File,
    header: str,
):
    
    header = format_vcf_header(
        header,
        chrom = None,
        add_filters = False
    )
        
    # Subset the columns in the VCF spec and add fields to INFO 
    vcf = vcf.rename(
        columns = {
            k: k.upper()
            for k in _VCF_HEADER
        }
    )

    # Convert DataFrame to TSV
    vcf_str = "\n".join(
        [
            "\t".join(x.astype(str)) 
            for x in vcf.values
        ]
    )

    # Write VCF
    with open(file, "w") as out:

        out.write(
            "\n".join(
                [
                    header,
                    vcf_str
                ]
            )
        )