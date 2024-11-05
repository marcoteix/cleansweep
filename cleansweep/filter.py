from dataclasses import dataclass
from typing import Union
from cleansweep.coverage import CoverageFilter
from cleansweep.basecounts import MAPClassifier
import pandas as pd

@dataclass
class VCFFilter:

    reference_ani: float = .998
    random_state: Union[int, None] = 23
    uncertainty: float = 5.0

    def fit(self, vcf: pd.DataFrame, expected_coverage: Union[int, None] = None, 
        coverage_filter_params: dict = {}, basecount_filter_params: dict = {}) -> pd.Series:
        """Fits the CleanSweep filters and predicts a filtering result (`PASS`, `FAIL`, or `HighCov`)
        for each variant in a Pilon VCF DataFrame.

        Args:
            vcf (pd.DataFrame): Pilon VCF DataFrame.
            expected_coverage (int | None, optional): Expected coverage of the query strain. If `None`, 
                estimates it from the component of lowest mean in the first coverage-based filter. 
                Defaults to None.
            coverage_filter_params (dict, optional): Parameters passed to the coverage filter. See 
                `CoverageFilter.fit()` for a list of parameters. Defaults to {}.
            basecount_filter_params (dict, optional): Parameters passed to the base counts filter. See 
                `MAPClassifier.fit()` for a list of parameters. Defaults to {}.

        Returns:
            filtering_results (pd.Series): Filtering results, with the same index as the input `vcf`.
        """

        # Fit and apply the coverage-based filter
        self.coverage_filter = CoverageFilter(random_state=self.random_state)
        filtered_vcf = self.coverage_filter.fit(vcf, **coverage_filter_params)

        # Fit the second filter (based on alt and ref base counts) with the 
        # passing sites from the first filter
        filtered_vcf = self.coverage_filter.filter(filtered_vcf)
        self.basecount_filter = MAPClassifier(reference_ani=self.reference_ani, uncertainty=self.uncertainty)
        self.basecount_filter.fit(filtered_vcf, **basecount_filter_params)

        # If there is no user provided expected coverage for the query strain,
        # estimate it using the coverage filter
        if expected_coverage is None:
            expected_coverage = self.coverage_filter.mean_query_coverage

        bc_probs = self.basecount_filter.predict_proba(filtered_vcf, 
            expected_coverage=expected_coverage)
        self.predictions = bc_probs.assign(
            cleansweep_filter=bc_probs.logp_pass.gt(bc_probs.logp_fail) \
                .replace({True: "PASS", False:"FAIL"}))
        
        # Join the final filter with the original VCF DataFrame
        self.predictions.cleansweep_filter = self.predictions.cleansweep_filter.fillna("HighCov")

        return self.predictions
    
    def fit_filter(self, vcf: pd.DataFrame, expected_coverage: Union[int, None] = None, 
        coverage_filter_params: dict = {}, basecount_filter_params: dict = {}) -> pd.DataFrame:
        """Fits the CleanSweep filters and filters the variants in a Pilon VCF DataFrame.

        Args:
            vcf (pd.DataFrame): Pilon VCF DataFrame.
            expected_coverage (int | None, optional): Expected coverage of the query strain. If `None`, 
                estimates it from the component of lowest mean in the first coverage-based filter. 
                Defaults to None.
            coverage_filter_params (dict, optional): Parameters passed to the coverage filter. See 
                `CoverageFilter.fit()` for a list of parameters. Defaults to {}.
            basecount_filter_params (dict, optional): Parameters passed to the base counts filter. See 
                `MAPClassifier.fit()` for a list of parameters. Defaults to {}.

        Returns:
            filtering_results (pd.DataFrame): Variants in the input `vcf` passing all CleanSweep filters.
        """

        filter_results = self.fit(vcf, expected_coverage=expected_coverage,
            coverage_filter_params=coverage_filter_params, 
            basecount_filter_params=basecount_filter_params)
        vcf = vcf.join(filter_results)
        return vcf[vcf.cleansweep_filter.eq("PASS")].drop(columns="cleansweep_filter")

    def summary(self) -> dict:
        """Get summary statistics from the last fit of this object.

        Returns:
            stats (dict): Sumary statistics.
        """

        if not hasattr(self, "predictions"):
            raise RuntimeError("This object does not have predictions. Please fit the \
filters (with fit() or fit_filter())before calling summary().")

        return {
            "Coverage filter (Step 1)":
                {
                    "Number of excluded variants": self.predictions.eq("HighCov").sum(),
                    "Number of passing variants": self.predictions.ne("HighCov").sum(),
                    "Means of the estimated components": 
                        ", ".join([str(x) for x in self.coverage_filter.get_means()])
                },
            "Base counts filter (Step 2)":
                {
                    "Number of excluded variants": self.predictions.eq("FAIL").sum(),
                    "Number of passing variants": self.predictions.eq("PASS").sum()
                }
        }