import matplotlib.pyplot as plt 
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from dataclasses import dataclass 
from typing import Tuple, Union
from abc import ABC, abstractmethod
from cleansweep.typing import File
from cleansweep.filter import VCFFilter
from typing_extensions import Self
import pandas as pd
import joblib

@dataclass
class Plot(ABC):

    figsize: Tuple[int, int] = (6,5)
    rows: int = 1
    columns: int = 1

    @abstractmethod
    def plot(
        self,
        ax: Union[plt.Axes, None]
    ) -> plt.Axes:

        sns.set_theme("talk")
        sns.set_style("whitegrid")

        # If no axes were provided, create
        if ax is None:
            _, ax = plt.subplots(
                self.rows,
                self.columns,
                constrained_layout = True,
                figsize = self.figsize
            )

        return ax 

    def save(
        self,
        path: File
    ) -> Self: 
        
        plt.savefig(path)
        plt.show()

        return self
    
    def config_axes(
        self,
        ax: plt.Axes, 
        *, 
        move_legend: bool = True,
        legend_title = None,  
        grid: bool = True, 
        xlabel = None, 
        ylabel = None, 
        xlog: bool = False, 
        ylog: bool = False,
        xrotation = None, 
        ypercent: bool = False, 
        title = None, 
        xlim = None, 
        ylim = None,
        xticks = None,
        xticklabels = None,
        yticks = None,
        yticklabels = None,
        despine: bool = True
    ):

        if despine: 
            sns.despine(ax=ax)

        if not ax.get_legend() is None: 
            if move_legend: 
                ax.legend()

            sns.move_legend(
                ax, 
                loc="center left", 
                bbox_to_anchor=(1, 0.5), 
                frameon=False, 
                title=legend_title
            )

        elif not ax.get_legend() is None:
            sns.move_legend(
                ax, 
                loc="best", 
                frameon=False, 
                title=legend_title
            )


        if grid: 
            ax.grid(alpha=.1)

        ax.set(
            ylabel = ylabel, 
            xlabel = xlabel, 
            title = title
        )

        if xlog: 
            ax.set_xscale("log")

        if ylog: 
            ax.set_yscale("log")

        if xticks is None:
            xticks = ax.get_xticks()
        if xticklabels is None:
            xticklabels = xticks 
        
        if yticks is None:
            yticks = ax.get_yticks()
        if yticklabels is None:
            yticklabels = yticks
        
        if xrotation: 
            ax.set_xticks(
                ax.get_xticks(), 
                ax.get_xticklabels(), 
                va="top", 
                ha="right" if xrotation != 90 else "center", 
                rotation=xrotation
            )

        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)

        if ypercent: 
            ax.yaxis.set_major_formatter(
                PercentFormatter(1, 0)
            )

        return ax
    


class AlleleDepthsPlot(Plot):

    def plot(
        self, 
        vcf: pd.DataFrame,
        *,
        log: bool = True,
        ax: Union[plt.Axes, None] = None
    ) -> plt.Axes:
        
        ax = super().plot(ax)

        self.__validate(vcf)

        sns.scatterplot(
            vcf,
            x = "ref_bc",
            y = "alt_bc",
            hue = "filter",
            palette = "Set1",
            alpha = .3,
            size = "p_alt",
            size_norm=(-.01,1),
            hue_order = [
                "FAIL",
                "PASS",
                "RefVar",
                "LowAltBC",
                "LowCov"
            ],
            ax = ax
        )

        self.config_axes(
            ax = ax,
            move_legend = True,
            legend_title = "CleanSweep\nfilter",
            xlabel = "Reference allele depth",
            ylabel = "Alternate allele depth",
            ylog = log,
            xlog = log,
            title = str(vcf.chrom.iloc[0])
        )

        return ax

    def __validate(
        self,
        vcf: pd.DataFrame
    ) -> Self:
        
        for attr in [
            "ref_bc",
            "alt_bc",
            "filter",
            "p_alt",
            "chrom"
        ]:
            if not attr in vcf:
                raise ValueError(
                    f"The VCF does not have a \"{attr}\" column."
                )
        
        if not len(vcf):
            raise ValueError(
                "Got an empty VCF."
            )
        
        return self
        
class QueryDepthsPlot(Plot): 

    def plot(
        self,
        cleansweep: File,
        *,
        log: bool = True,
        bins: Union[None, int] = None,
        title: str = "",
        ax: Union[plt.Axes, None] = None
    ) -> plt.Axes:
        
        # Load CleanSweep filter
        cleansweep_obj = joblib.load(
            cleansweep
        )
        
        ax = super().plot(ax)

        sns.histplot( 
            pd.Series(
                    cleansweep_obj.coverage_estimator.depths
                ).rename("depths") \
                .to_frame(),
            x = "depths",
            bins = bins if bins else 10,
            color = ".3",
            log_scale = log,
            ax = ax
        )

        ax.vlines(
            cleansweep_obj.query_coverage,
            *ax.get_ylim(),
            ls = "--",
            color = ".8",
            label = "Estimated mean\nquery depth\nof coverage"
        )
    
        self.config_axes(
            ax = ax,
            move_legend = False,
            xlabel = "Depth of coverage",
            ylabel = "Number of sites",
            title = title
        )

        return ax