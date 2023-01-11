import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.timeseries.base import TimeSeriesMetric
from sdmetrics.timeseries.utils import distribution_similarity


class InterarrivalDistSimilarity(TimeSeriesMetric):
    name = "Interarrival distributional similarity"
    goal = Goal.MINIMIZE

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None,
                entity_columns=None):
        _, entity_columns = cls._validate_inputs(
            real_data, synthetic_data, metadata, entity_columns)
        attribute_cols = metadata['entity_columns'] + \
            metadata['context_columns']

        column_sequence_index = metadata["sequence_index"]
        real_data[column_sequence_index] = pd.to_datetime(
            real_data[column_sequence_index]).astype(int) / 10**9
        synthetic_data[column_sequence_index] = pd.to_datetime(
            synthetic_data[column_sequence_index]).astype(int) / 10**9

        real_gk = real_data.groupby(attribute_cols)
        real_interarrival_within_flow_list = []

        synthetic_gk = synthetic_data.groupby(attribute_cols)
        synthetic_interarrival_within_flow_list = []

        max_length = max(max(real_gk.size()), max(synthetic_gk.size()))

        for group_name, df_group in real_gk:
            real_interarrival_within_flow_list.append([0.0] +
                                                      list(np.diff(df_group[column_sequence_index])) +
                                                      [0.0] * (max_length - len(df_group)))

        real_interarrival_within_flow_list = np.asarray(
            real_interarrival_within_flow_list).reshape(-1, max_length)

        for group_name, df_group in synthetic_gk:
            synthetic_interarrival_within_flow_list.append([0.0] +
                                                           list(np.diff(df_group[column_sequence_index])) +
                                                           [0.0] * (max_length - len(df_group)))

        synthetic_interarrival_within_flow_list = np.asarray(
            synthetic_interarrival_within_flow_list).reshape(-1, max_length)

        column_names = [f"interarrival_{i}" for i in range(max_length)]
        date_type = ['numerical' for i in range(max_length)]

        scores = {}
        if np.array_equal(real_interarrival_within_flow_list,
                          synthetic_interarrival_within_flow_list):
            scores["interarrival"] = 0.0
        else:
            scores["interarrival"] = distribution_similarity(
                real_data=real_interarrival_within_flow_list,
                synthetic_data=synthetic_interarrival_within_flow_list,
                column_names=column_names,
                data_type=date_type,
                comparison_type='both',
                categorical_mapping=True
            )
        return scores
