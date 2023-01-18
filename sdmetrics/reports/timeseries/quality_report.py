"""Timeseries quality report"""
import sys
import pickle
import random
import warnings
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

METRICS = [
    SingleAttrDistSimilarity,
    SingleAttrCoverage,
    SessionLengthDistSimilarity,
    FeatureDistSimilarity,
    CrossFeatureCorrelation,
    InterarrivalDistSimilarity,
    PerFeatureAutocorrelation,
    SingleAttrSingleFeatureCorrelation
]


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
        for metric in METRICS:
            out.write("="*80+"\n")
            out.write(f"Metric: {metric.name}\n")
            try:
                self._print_scores(metric.compute(
                    real_data, synthetic_data, metadata), out)
            except:
                attribute_cols = metadata['entity_columns'] + metadata['context_columns']
                feature_cols = list(set(real_data.columns) -
                                    set(attribute_cols))
                if metric == CrossFeatureCorrelation:
                    self._print_scores(
                        metric.compute(
                            real_data, synthetic_data, metadata,
                            target=random.choices(
                                [f for f in feature_cols
                                 if metadata['fields'][f]['type']
                                 == 'numerical'],
                                k=2)), out)

                elif metric == PerFeatureAutocorrelation:
                    self._print_scores(
                        metric.compute(
                            real_data, synthetic_data, metadata,
                            target=random.choice(
                                [f for f in feature_cols
                                 if metadata['fields'][f]['type']
                                 == 'numerical'])), out)

                elif metric == SingleAttrSingleFeatureCorrelation:
                    self._print_scores(
                        metric.compute(real_data, synthetic_data, metadata,
                                       attr_name=random.choice(
                                           [f for f in attribute_cols
                                            if metadata['fields'][f]['type']
                                               == 'categorical']),
                                       feature_name=random.choice(
                                           [f for f in feature_cols
                                            if metadata['fields'][f]['type']
                                            == 'numerical'])
                                       ), out)

                else:
                    out.write("Metric is not compatible with this dataset.\n")
            out.write("="*80+"\n\n")

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

    @classmethod
    def load(cls, filepath):
        """Load a ``QualityReport`` instance from a given path.

        Args:
            filepath (str):
                The path to the file where the report is stored.

        Returns:
            QualityReort:
                The loaded quality report instance.
        """
        current_version = pkg_resources.get_distribution('sdmetrics').version

        with open(filepath, 'rb') as f:
            report = pickle.load(f)
            if current_version != report._package_version:
                warnings.warn(
                    f'The report was created using SDMetrics version `{report._package_version}` '
                    f'but you are currently using version `{current_version}`. '
                    'Some features may not work as intended.')

            return report
