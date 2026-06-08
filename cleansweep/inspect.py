from dataclasses import dataclass
from typing import List, Union
from cleansweep.filter import VCFFilter
from cleansweep.typing import File
import joblib
import numpy as np
import pandas as pd
from scipy import stats as sps

@dataclass
class Inspector:

    def vcf_stats(
        self,
        vcf = pd.DataFrame,
        header: Union[str, None] = None
    ) -> dict:

        stats = {}

        # Number of variants
        stats[
            "Number of variants"
        ] = len(vcf)

        # Number of multiallelic sites and nucleotide diversity (microdiversity)
        stats[
            "Number of multiallelic sites"
        ] = self.count_multiallelic_sites(vcf)

        stats[
            "Nucleotide diversity (pi)"
        ] = self.nucleotide_diversity(vcf, header)

        # Number of variants per CleanSweep filter
        stats[
            "Number of variants per CleanSweep filter"
        ] = vcf["filter"] \
            .value_counts() \
            .to_dict()
        
        passing = vcf[ 
            vcf["filter"].eq("PASS")
        ]

        stats[
            "Statistics about passing sites"
        ] = {
            "Depth of coverage (pileups)": passing.depth.describe().to_dict(),
            "Reference allele depth": passing.ref_bc.describe().to_dict(),
            "Alternate allele depth": passing.alt_bc.describe().to_dict()
        }

        failing = vcf[ 
            vcf["filter"].ne("PASS")
        ]

        stats[
            "Statistics about excluded sites"
        ] = {
            "Depth of coverage (pileups)": failing.depth.describe().to_dict(),
            "Reference allele depth": failing.ref_bc.describe().to_dict(),
            "Alternate allele depth": failing.alt_bc.describe().to_dict()
        }        

        stats[
            "Statistics about the estimated probability of each variant:"
        ] = vcf.p_alt.describe().to_dict()

        return stats

    def count_multiallelic_sites(
        self,
        vcf: pd.DataFrame
    ) -> int:
        """Counts the number of distinct sites called multiallelic by CleanSweep.

        Multiallelic sites are written as several rows (one alternate allele per row)
        carrying the MULTI flag, parsed back into the ``is_multiallelic`` column.
        """

        if "is_multiallelic" not in vcf.columns:
            return 0

        multiallelic = vcf[vcf["is_multiallelic"] == True]

        return int(
            multiallelic.drop_duplicates(["chrom", "pos"]).shape[0]
        )

    def nucleotide_diversity(
        self,
        vcf: pd.DataFrame,
        header: Union[str, None] = None
    ) -> float:
        """Estimates the nucleotide diversity (pi) of the target strain.

        For each site, the per-site diversity is the expected number of pairwise
        differences, ``1 - sum_i f_i**2``, where ``f_i`` are the estimated allele
        fractions, weighted by the probability that the site is multiallelic. The
        genome-wide diversity is the sum over sites divided by the genome length
        (parsed from the VCF ``##contig`` header lines). Monoallelic sites contribute
        zero, so only sites with multiple co-existing alleles add to pi.

        Args:
            vcf (pd.DataFrame): Output VCF read without collapsing multiallelic lines,
                with ``allele_fraction`` and ``p_multi`` columns.
            header (Union[str, None]): VCF header text, used to read the genome length.

        Returns:
            float: Estimated nucleotide diversity (pi).
        """

        if "allele_fraction" not in vcf.columns or "p_multi" not in vcf.columns:
            return 0.0

        evaluated = vcf[
            vcf["allele_fraction"].notna() & vcf["p_multi"].notna()
        ]

        if not len(evaluated):
            return 0.0

        pi_total = 0.0
        for _, site in evaluated.groupby(["chrom", "pos"]):
            alt_fractions = site["allele_fraction"].to_numpy(dtype=float)
            ref_fraction = max(0.0, 1.0 - alt_fractions.sum())
            fractions = np.append(alt_fractions, ref_fraction)
            p_multi = float(site["p_multi"].iloc[0])
            pi_total += p_multi * (1.0 - np.sum(fractions ** 2))

        genome_length = self.genome_length(header, vcf)

        return pi_total / genome_length

    def genome_length(
        self,
        header: Union[str, None],
        vcf: pd.DataFrame
    ) -> int:
        """Returns the total genome length from the VCF ``##contig`` header lines.

        Falls back to the number of distinct evaluated sites when the contig lengths are
        not available in the header.
        """

        chroms = list(pd.unique(vcf["chrom"])) if len(vcf) else []
        total = 0

        if header is not None:
            for chrom in chroms:
                marker = f"##contig=<ID={chrom},length="
                if marker in header:
                    try:
                        total += int(header.split(marker)[-1].split(">")[0].split(",")[0])
                    except (ValueError, IndexError):
                        pass

        if total > 0:
            return total

        # Fallback: number of distinct sites that were evaluated
        return max(1, vcf.drop_duplicates(["chrom", "pos"]).shape[0])

    def cleansweep_info(
        self,
        cleansweep: VCFFilter
    ) -> dict:

        attrs = {
            "Fitting method": cleansweep.method,
            "Random seed": cleansweep.random_state,
            "Estimated mean query depth of coverage": cleansweep.query_coverage,
        }

        # Derive NB r/p from whichever method was used
        if cleansweep.method == "fast":
            r = cleansweep.coverage_estimator.r
            p = cleansweep.coverage_estimator.p
        else:
            # mixture: pyMC parametrises NB as (mu, alpha).
            # scipy parametrises as (n, p) with n = alpha, p = alpha/(alpha+mu)
            alpha = cleansweep.basecount_filter.dist_params["query_overdispersion"]
            mu = cleansweep.query_coverage
            r = alpha
            p = alpha / (alpha + mu)

            attrs["Estimated overdispersion for the query depths of coverage "
                  "(query_overdispersion) and probability of a variant being "
                  "true (alt_prob)"] = cleansweep.basecount_filter.dist_params
            attrs["MCMC options"] = {
                "Number of chains": cleansweep.basecount_filter.chains,
                "Number of draws per chain": cleansweep.basecount_filter.draws,
                "Number of burn-in draws per chain": cleansweep.basecount_filter.burn_in,
                "Engine": cleansweep.basecount_filter.engine,
                "Number of threads": cleansweep.basecount_filter.threads
            }

        dist = sps.nbinom(r, p)
        attrs["Negative Binomial parameters"] = {
            "r": r,
            "p": p,
            "mean": float(dist.mean()),
            "std": float(dist.std()),
            "percentile_2.5": float(dist.ppf(0.025)),
            "percentile_97.5": float(dist.ppf(0.975)),
        }

        return attrs
    
    def report(
        self,
        vcf: pd.DataFrame,
        cleansweep: File,
        header: Union[str, None] = None
    ) -> dict:

        cleansweep_filter = self.load_filter(cleansweep)
        return {
            "VCF statistics": self.vcf_stats(vcf, header=header),
            "CleanSweep filter information": self.cleansweep_info(cleansweep_filter)
        }


    def load_filter(
        self,
        file: File
    ) -> VCFFilter:

        return joblib.load(
            file
        )