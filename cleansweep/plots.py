#%%
import matplotlib.pyplot as plt 
from matplotlib.ticker import PercentFormatter
import numpy as np
import seaborn as sns
from dataclasses import dataclass 
from typing import Literal, Tuple, Union
from abc import ABC, abstractmethod
from cleansweep.typing import File
from typing_extensions import Self
from cleansweep_mcmc import SamplingResult
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
                    cleansweep_obj["query_coverage_estimator"].depths
                ).rename("depths") \
                .to_frame(),
            x = "depths",
            bins = bins if bins else 10,
            color = ".3",
            log_scale = log,
            ax = ax
        )

        ax.vlines(
            cleansweep_obj["query_coverage"],
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
    
class TracePlot(Plot):

    def plot(
        self,
        posterior: dict,
        parameter: Literal["alleles", "alt_allele_proportion", "dispersion"],
        *,
        palette: str = "Set1",
        title: str = "",
        ax: Union[plt.Axes, None] = None,
        **kwargs
    ) -> plt.Axes:
        
        ax = super().plot(ax)
        
        colors = sns.color_palette(
            palette,
            len(posterior[parameter])
        )

        for chain, y in posterior[parameter].items():

            n_samples = len(y)
            
            if parameter != "alleles":

                ax.plot(
                    np.arange(n_samples) + 1,
                    y,
                    label = f"Chain {chain+1}",
                    lw = 1,
                    color = colors[chain],
                    **kwargs
                )

                ax.legend()

                self.config_axes(
                    ax = ax,
                    move_legend = True,
                    xlabel = "Iteration",
                    ylabel = "Value",
                    title = title
                )

            else:

                # Plot heatmap
                ax.imshow(
                    y.transpose(),
                    cmap = "gray",
                    interpolation = "nearest",
                    aspect = "auto"
                )

                self.config_axes(
                    ax = ax,
                    move_legend = False,
                    xlabel = "Iteration",
                    ylabel = "Allele",
                    title = title
                )

        return ax
    
class PosteriorPlot(Plot):

    def plot(
        self,
        posterior: dict,
        parameter: Literal["alleles", "alt_allele_proportion", "dispersion"],
        *,
        show_chains: bool = True,
        title: str = "",
        palette: str = "Set1",
        ax: Union[plt.Axes, None] = None,
        hist_kwargs: dict = {},
        kde_kwargs: dict = {},
        bar_kwargs: dict = {}
    ) -> plt.Axes:
        
        ax = super().plot(ax)
        
        if parameter == "alleles":

            # Reshape to n_samples x n_alleles            
            values = {
                chain + 1: pd.Series(
                    x.mean(axis=0)
                )
                for chain, x in posterior[parameter].items()
            }

            values = pd.concat(
                values,
                names = ["chain", "allele"]
            ).rename("value") \
            .to_frame() \
            .reset_index()

            if not show_chains:

                values = values.groupby(
                    "allele",
                    as_index = False
                ).mean()

                palette = None

            sns.barplot(
                values,
                x = "allele",
                y = "value",
                hue = (
                    "chain"
                    if show_chains
                    else None
                ),
                color = (
                    "0.3"
                    if not show_chains
                    else None
                ),
                palette = palette,
                lw = 0,
                ax = ax,
                **bar_kwargs
            )

            self.config_axes(
                ax = ax,
                move_legend = True,
                legend_title = "Chain",
                xlabel = "Site",
                ylabel = "Probability",
                title = title
            )

            ax.set_xticks([])

        else:

            values = {
                chain+1: pd.Series(x)
                for chain, x in posterior[parameter].items()
            }

            values = pd.concat(
                values,
                names = ["chain", "iteration"]
            ).rename("value") \
            .to_frame() \
            .reset_index()

            sns.histplot(
                values,
                x = "value",
                hue = (
                    "chain"
                    if show_chains
                    else None
                ),
                palette = palette,
                stat = "density",
                ax = ax,
                **hist_kwargs
            )

            sns.kdeplot(
                values,
                x = "value",
                hue = (
                    "chain"
                    if show_chains
                    else None
                ),
                palette = palette,
                ax = ax,
                **kde_kwargs
            )

            self.config_axes(
                ax = ax,
                move_legend = True,
                legend_title = "Chain",
                xlabel = "Value",
                ylabel = "Probability",
                xlim = (0, 1),
                title = title
            )

        return ax
    
class AutocorrelationPlot(Plot):

    def plot(
        self,
        posterior: dict,
        parameter: Literal["alt_allele_proportion", "dispersion"],
        *,
        title: str = "",
        palette: str = "Set1",
        max_lag: Union[int, None] = None,
        ax: Union[plt.Axes, None] = None,
        **kwargs        
    ) -> plt.Axes:
        
        ax = super().plot(ax)

        colors = sns.color_palette(
            palette,
            len(posterior[parameter])
        )

        for chain, samples in posterior[parameter].items():

            values = pd.Series(samples)
            n_samples = len(values)

            if max_lag is None: max_lag = n_samples-1
            max_lag = np.minimum(n_samples-1, max_lag)

            # Get autocorrelation at different lags
            y = [values.autocorr(k) for k in range(max_lag)]

            ax.plot(
                np.arange(max_lag),
                y,
                color = colors[chain],
                alpha = .8,
                label = f"Chain {chain + 1}",
                **kwargs
            )

        self.config_axes(
            ax = ax,
            move_legend = True,
            legend_title = "Chain",
            xlabel = "Lag",
            ylabel = "Autocorrelation",
            ylim = (0, 1),
            title = title
        )

        return ax
# %%
