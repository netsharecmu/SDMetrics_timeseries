import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.timeseries.base import TimeSeriesMetric
from sdmetrics.timeseries.utils import autocorrelation_similarity


class PerFeatureAutocorrelation(TimeSeriesMetric):
    name = "PerFeatureAutocorrelation"
    goal = Goal.MINIMIZE

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None,
                entity_columns=None, target=None):
        _, entity_columns = cls._validate_inputs(
            real_data, synthetic_data, metadata, entity_columns)

        assert isinstance(target, str), \
            "target should be a string type"

        assert target in real_data.columns, \
            "target should exists as a column name in real data"

        assert target in synthetic_data.columns, \
            "target should exists as a column name in synthetic data"

        attribute_cols = metadata['entity_columns'] + \
            metadata['context_columns']

        real_gk = real_data.groupby(attribute_cols)
        real_feature = []

        synthetic_gk = synthetic_data.groupby(attribute_cols)
        synthetic_feature = []

        max_length = max(max(real_gk.size()), max(synthetic_gk.size()))

        for group_name, df_group in real_gk:
            real_feature.append(list(df_group[target]) +
                                [0.0] * (max_length - len(df_group)))

        real_feature = np.asarray(
            real_feature).reshape(-1, max_length)

        for group_name, df_group in synthetic_gk:
            synthetic_feature.append(list(df_group[target]) +
                                     [0.0] * (max_length - len(df_group)))

        synthetic_feature = np.asarray(
            synthetic_feature).reshape(-1, max_length)

        column_names = [target]
        data_type = ["numerical"]
        comparison_type = "both"

        scores = {}
        scores[f"autocorrelation_{target}"] = autocorrelation_similarity(
            real_data=real_feature,
            synthetic_data=synthetic_feature,
            column_names=column_names,
            data_type=data_type,
            comparison_type=comparison_type
        )
        return scores
