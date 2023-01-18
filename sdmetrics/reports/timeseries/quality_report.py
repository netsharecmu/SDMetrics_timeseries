"""Timeseries quality report"""
import sys
import pickle
import pkg_resources

from tqdm import tqdm

from sdmetrics.timeseries import TimeSeriesMetric
from sdmetrics.timeseries.fidelity import (
    SingleAttrDistSimilarity,
    SingleAttrCoverage,
    SessionLengthDistSimilarity,
    FeatureDistSimilarity,
    CrossFeatureCorrelation,
    InterarrivalDistSimilarity,
    PerFeatureAutocorrelation,
    SingleAttrSingleFeatureCorrelation
)


class QualityReport():
    def _print_scores(self, scores, out):
        for col, score in scores.items():
            assert len(score) >= 1, \
                "At least numerical score has to be generated"

            out.write(f"Column: {col}\n")
            out.write(f"Numeric score: {score[0]}\n")
            # Display figure
            if len(score) == 2:
                score[1].show()

    def generate(self, real_data, synthetic_data, metadata, out=sys.stdout):
        # self._print_scores(
        #     SingleAttrDistSimilarity.compute(
        #         real_data, synthetic_data, metadata), out)
        # self._print_scores(
        #     SingleAttrCoverage.compute(
        #         real_data, synthetic_data, metadata), out)
        # self._print_scores(
        #     SessionLengthDistSimilarity.compute(
        #         real_data, synthetic_data, metadata), out)
        # self._print_scores(
        #     FeatureDistSimilarity.compute(
        #         real_data, synthetic_data, metadata), out)
        self._print_scores(
            CrossFeatureCorrelation.compute(
                real_data, synthetic_data, metadata=metadata,
                target=['total_sales', 'nb_customers']),
            out)

    def save(self, filepath):
        """Save this report instance to the given path using pickle.

        Args:
            filepath (str):
                The path to the file where the report instance will be serialized.
        """
        self._package_version = pkg_resources.get_distribution(
            'sdmetrics').version

        with open(filepath, 'wb') as output:
            pickle.dump(self, output)
