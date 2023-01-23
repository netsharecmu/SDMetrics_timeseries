import numpy as np
import pandas as pd

from typing import List

from sdmetrics.goal import Goal
from sdmetrics.timeseries.base import TimeSeriesMetric
from sdmetrics.timeseries.utils import distribution_similarity


class AttrDistSimilarity(TimeSeriesMetric):
    """This compares the distribution of a single or multiple metadata attributes (e.g., device type) between the real and synthetic data. It computes a distance between the two distributions."""

    name = "Attribute distributional similarity"
    goal = Goal.MINIMIZE

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None,
                entity_columns=None, target=None):
        if not all(isinstance(s, str) for s in target):
            raise ValueError(
                "target has to be a list of strings where each string specifies an attribute column.")

        _, entity_columns = cls._validate_inputs(
            real_data, synthetic_data, metadata, entity_columns)
        real_data_attribute, _, _ = \
            cls._load_attribute_feature(real_data, metadata)
        synthetic_data_attribute, _, _ = \
            cls._load_attribute_feature(synthetic_data, metadata)
        attribute_cols, feature_cols = cls._get_attribute_feature_cols(metadata)
        for col in target:
            if col not in attribute_cols:
                raise ValueError(f"Column {col} is not an attribute.")

        real_columns = real_data[target].to_numpy().reshape(-1, len(target))
        synthetic_columns = synthetic_data[target].to_numpy(
        ).reshape(-1, len(target))

        return distribution_similarity(
            real_data=real_columns,
            synthetic_data=synthetic_columns,
            column_names=target,
            data_type=[metadata['fields'][col]['type'] for col in target],
            comparison_type='both',
            categorical_mapping=True
        )
