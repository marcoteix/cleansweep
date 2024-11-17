#%%
import arviz as az
from dataclasses import dataclass

import numpy as np
from cleansweep.basecounts import BaseCountEstimator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matiss import plots
plots.set_font("Arial")
sns.set_style("whitegrid")
sns.set_context("talk")
sns.set_palette("Set1")

@dataclass
class Diagnostics:

    base_count_estimator: BaseCountEstimator

    def __post_init__(self):

        self.samples = self.base_count_estimator.sampling_results

    def trace(self, vars_ = None):

        az.plot_trace(self.samples, var_names=vars_)

    def energy(self, **kwargs):

        az.plot_energy(self.samples, **kwargs)
    
    def plot_errors(self, scores: pd.Series, gt: pd.Series, 
        data: pd.DataFrame, bias: float=0.5):

        predictions = scores.ge(bias)
        X = data.join(predictions.rename("prediction")) \
            .join(gt.eq("TP").rename("GT")) \
            .join(scores.rename("p_true"))
        X = X.assign(result=
            X.GT.eq(X.prediction).replace({True: "T", False: "F"}) + X \
                .prediction.replace({True: "P", False: "N"}))

        fig, ax = plots.get_figure(figsize=(7,7))
        sns.scatterplot(X, x="alt_bc", y="ref_bc", hue="result", 
            alpha=.5, ax=ax, size="p_true")
        xmax = np.percentile(X.alt_bc, 99)
        ymax = np.percentile(X.ref_bc, 99)
        plots.config_axes(ax, legend_title="CleanSweep Score", 
            xlabel="Alternate allele base count",
            ylabel="Reference allele base count",
            xlim=(0,xmax), ylim=(0,ymax)
        )

        plt.show()
# %%