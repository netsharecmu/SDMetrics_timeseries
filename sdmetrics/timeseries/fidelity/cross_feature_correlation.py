import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.timeseries.base import TimeSeriesMetric
from sdmetrics.timeseries.utils import pearson_corr


class CrossFeatureCorrelation(TimeSeriesMetric):
    """This metric computes the Pearson correlation between two features f1 and f2. This correlation is computed (averaged) over every record in every time series in the (real or synthetic) dataset. We then compare the difference in these correlation values between the real and synthetic datasets."""

    name = "Cross Feature Correlation"
    goal = Goal.MAXIMIZE

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None,
                entity_columns=None, target=None):
        _, entity_columns = cls._validate_inputs(
            real_data, synthetic_data, metadata, entity_columns)

        assert isinstance(target, list), \
            "target should be a list type"

        assert len(target) == 2, \
            "expected is expected to be a list including two elements representing two columns."

        column_1 = target[0]
        column_2 = target[1]
        assert column_1 in metadata['fields'], \
            f"column {column_1} should exist in the dataframe"
        assert column_2 in metadata['fields'], \
            f"column {column_2} should exist in the dataframe"
        assert metadata['fields'][column_1]['type'] in ['numerical'], \
            f"column {column_1} should be numerical"
        assert metadata['fields'][column_2]['type'] in ['numerical'], \
            f"column {column_2} should be numerical"
        scores = {}

        real_corr = pearson_corr(real_data[column_1].to_numpy(
        ), real_data[column_2].to_numpy())
        synthetic_corr = pearson_corr(
            synthetic_data[column_1].to_numpy(),
            synthetic_data[column_2].to_numpy())

        scores[str(target)] = [1 - (abs(real_corr - synthetic_corr)) / 2]
        return scores
