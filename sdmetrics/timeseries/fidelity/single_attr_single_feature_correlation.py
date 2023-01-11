import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.timeseries.base import TimeSeriesMetric


class SingleAttrSingleFeatureCorrelation(TimeSeriesMetric):
    """A metadata attribute a can take different values v. The generated time series for a feature f may depend on the value v of metadata attribute a. This metric checks that correlation qualitatively by fixing the value v, and comparing the distribution of the time series for feature f between real and synthetic data. """

    name = "Correlation between a metadata attribute a and a time series feature f"
    goal = Goal.MINIMIZE

    def compute(cls, real_data, synthetic_data, metadata=None,
                entity_columns=None):
        _, entity_columns = cls._validate_inputs(
            real_data, synthetic_data, metadata, entity_columns)
        real_data_attribute, real_data_feature, real_data_gen_flag = \
            cls._load_attribute_feature(real_data, metadata)
        synthetic_data_attribute, _, _ = \
            cls._load_attribute_feature(synthetic_data, metadata)
