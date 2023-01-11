import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.timeseries.base import TimeSeriesMetric
from sdmetrics.timeseries.utils import coverage


class SingleAttrCoverage(TimeSeriesMetric):
    name = "Single attribute coverage"
    goal = Goal.MAXIMIZE

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None,
                entity_columns=None):
        _, entity_columns = cls._validate_inputs(
            real_data, synthetic_data, metadata, entity_columns)
        real_data_attribute, _, _ = \
            cls._load_attribute_feature(real_data, metadata)
        synthetic_data_attribute, _, _ = \
            cls._load_attribute_feature(synthetic_data, metadata)
        scores = {}
        for column_name, real_column in real_data.items():
            if column_name not in metadata['context_columns']:
                continue
            if column_name in metadata['fields']:
                real_column = real_column.to_numpy().reshape(-1, 1)
                synthetic_column = synthetic_data[column_name].to_numpy(
                ).reshape(-1, 1)

                scores[column_name] = coverage(
                    real_data=real_column,
                    synthetic_data=synthetic_column,
                    column_names=[column_name],
                    data_type=[metadata['fields'][column_name]['type']],
                    comparison_type='both'
                )

        return scores
