#%%
import matplotlib.pyplot as plt 
from cleansweep.filter import VCFFilter
import seaborn as sns
from typing import Tuple, Union
from itertools import permutations
import pandas as pd
import numpy as np

def plot_posterior(
    vcf_filter: VCFFilter,
    basecount_range: Tuple[int, int] = (0, 500),
    basecount_step: int = 1,
    figsize: Tuple[float, float] = (7,5)
):
    
    # Get a grid of alt and ref allele base counts
    bc_grid = pd.DataFrame(
        np.array(
            list(
                permutations(np.arange(*basecount_range, basecount_step), 2)
            )
        ),
        columns = ["alt_bc", "ref_bc"]
    )

    # Compute the posterior
    posteriors = vcf_filter.basecount_filter.get_posterior(
        observed = bc_grid,
        sampling_results = vcf_filter.basecount_filter.sampling_results,
        ambiguity_factor = vcf_filter.basecount_filter.ambiguity_factor,
        query_coverage = None
    ).rename("posterior")

    X = bc_grid.join(posteriors)

    fig, ax = plt.subplots(
        1, 1,
        constrained_layout = True,
        figsize = figsize
    )
    sns.scatterplot(
        data=X, 
        x="alt_bc", 
        y="ref_bc", 
        hue="posterior",
        size = "posterior",
        palette="inferno",
        alpha = .3,
        ax=ax
    )
    ax.set_xlabel("Alternate allele base count")
    ax.set_ylabel("Reference allele base count")
    ax.grid(False)
    sns.despine(ax=ax)

    plt.show()


# %%
