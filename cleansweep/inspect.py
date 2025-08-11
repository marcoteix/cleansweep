from dataclasses import dataclass 
from cleansweep.vcf import VCF
from cleansweep.filter import VCFFilter
from cleansweep.typing import File, Directory
from cleansweep import plots
from datetime import datetime
from typing import Union, Iterable
from typing_extensions import Self
import joblib
import pandas as pd
import json

@dataclass
class Inspector:

    filter_dict: File
    outdir: Directory
    prefix: str = "cleansweep"
    vcf: Union[File, None] = None 

    def __post_init__(self):

        # Read dict with filter options and results
        self.__filter_dict = joblib.load(self.filter_dict)

        # Create a JSON report 
        self.__report = self.outdir.joinpath(self.prefix + ".info.json")

        self.__json = {
            "Time": datetime.now().astimezone().strftime("%d/%m/%Y %H:%M:%S (%Z)"),
            "SWP file": str(self.filter_dict),
            "CleanSweep filter options": self.__filter_dict["opts"]
        }

        # Add coverage info
        self.__json["Coverage estimation"] = self.coverage_estimation_info()

        # Add MCMC info
        self.__json["Allele depth filter (MCMC)"] = self.mcmc_info()

        # Add VCF stats
        if not self.vcf is None:
            self.__vcf = VCF(self.vcf).read(
                chrom = None
            ) 

            self.__json["Filtering results"] = self.vcf_stats(self.__vcf)


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
    
    def mcmc_info(self) -> dict:

        return {
            "Estimated parameters of the allele depth distribution": self.__filter_dict["basecount_filter"]["estimates"],
            "Acceptance rates": self.__filter_dict["basecount_filter"]["acceptance_rate"],
            "R-hat": self.__filter_dict["basecount_filter"]["rhat"],
            "Used MLE estimator?": self.__filter_dict["mle"]
        }
    
    def coverage_estimation_info(self) -> dict:

        return {
            "Estimated mean depth of coverage for the query": self.__filter_dict["query_coverage"]
        }

    def write_json(self) -> Self:

        with open(self.__report, "w") as file:
            json.dump(
                self.__json, 
                file,
                indent = 4
            )

        return self

    def plot_allele_depths(
        self,
        figsize = (7,5),
        logscale: bool = True,
        formats: Iterable[str] = ["png"]
    ) -> Self:
        
        ad_plot = plots.AlleleDepthsPlot(
            figsize = figsize
        )

        if self.vcf is None:
            raise ValueError("Please supply a VCF file to plot allele depths.")
        
        ad_plot.plot(
            vcf = self.__vcf,
            log = logscale
        )

        for fmt in formats:
            ad_plot.save(
                self.outdir.joinpath(
                    self.prefix + ".allele_depths." + fmt
                )
            )

        return self

    def plot_query_depths(
        self,
        figsize = (7,5),
        logscale: bool = True,
        formats: Iterable[str] = ["png"]        
    ) -> Self:
        
        qd_plot = plots.QueryDepthsPlot(
            figsize = figsize
        )

        qd_plot.plot(
            self.filter_dict,
            log = logscale,
            title = self.prefix + " depth of coverage"
        )

        for fmt in formats:
            qd_plot.save(
                self.outdir.joinpath(
                    self.prefix + ".query_depth." + fmt
                )
            )

        return self
    
    def plot_trace(
        self, 
        figsize = (7,5),
        formats: Iterable[str] = ["png"]
    ) -> Self:

        for param in [
            "alleles",
            "alt_allele_proportion",
            "dispersion"
        ]:
            
            trace_plot = plots.TracePlot(figsize)
            ax = trace_plot.plot(
                self.__filter_dict["basecount_filter"]["posterior"],
                param,
                title = param
            )

            if param != "alleles": ax.set_ylim(0, 1)

            for fmt in formats:
                trace_plot.save(
                    self.outdir.joinpath(
                        self.prefix + ".trace." + param + "." + fmt
                    )
                )

        return self

    def plot_autocorrelation(
        self,
        figsize = (7,5),
        formats: Iterable[str] = ["png"]
    ) -> Self:
        
        for param in [
            "alt_allele_proportion",
            "dispersion"
        ]:
            
            acorr_plot = plots.AutocorrelationPlot(figsize)

            acorr_plot.plot(
                self.__filter_dict["basecount_filter"]["posterior"],
                param,
                title = param,
                max_lag = int(self.__filter_dict["opts"]["Number of MCMC draws"]/2)
            )

            for fmt in formats:
                acorr_plot.save(
                    self.outdir.joinpath(
                        self.prefix + ".autocorrelation." + param + "." + fmt
                    )
                )

        return self
    
    def plot_posterior(
        self,
        figsize = (7,5),
        formats: Iterable[str] = ["png"]
    ) -> Self:
        
        for param in [
            "alt_allele_proportion",
            "dispersion",
            "alleles"
        ]:
            
            post_plot = plots.PosteriorPlot(figsize)

            post_plot.plot(
                self.__filter_dict["basecount_filter"]["posterior"],
                param,
                title = param,
            )

            for fmt in formats:
                post_plot.save(
                    self.outdir.joinpath(
                        self.prefix + ".posterior." + param + "." + fmt
                    )
                )

        return self