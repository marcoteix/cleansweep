from dataclasses import dataclass
from cleansweep.filter import VCFFilter
from cleansweep.typing import File
import joblib
import pandas as pd
from scipy import stats as sps

@dataclass
class Inspector:

    def vcf_stats(
        self,
        vcf = pd.DataFrame
    ) -> dict:
        
        stats = {}

        # Number of variants
        stats[
            "Number of variants"
        ] = len(vcf)

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
        cleansweep: File
    ) -> dict:

        cleansweep_filter = self.load_filter(cleansweep)
        return {
            "VCF statistics": self.vcf_stats(vcf),
            "CleanSweep filter information": self.cleansweep_info(cleansweep_filter)
        }


    def load_filter(
        self,
        file: File
    ) -> VCFFilter:

        return joblib.load(
            file
        )