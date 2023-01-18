"""Timeseries quality report"""
import sys
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

            if len(score) == 2:
                score[1].show()

    def generate(self, real_data, synthetic_data, metadata, out=sys.stdout):
        self._print_scores(
            SingleAttrDistSimilarity.compute(
                real_data, synthetic_data, metadata), out)
