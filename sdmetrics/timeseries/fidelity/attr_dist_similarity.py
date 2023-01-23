import numpy as np
import pandas as pd

from typing import List

from sdmetrics.goal import Goal
from sdmetrics.timeseries.base import TimeSeriesMetric
from sdmetrics.timeseries.utils import distribution_similarity


class AttrDistSimilarity(TimeSeriesMetric):
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

        scores = {}
        for column_name, real_column in real_data.items():
            if column_name not in metadata['entity_columns'] + metadata['context_columns']:
                continue
            if column_name in metadata['fields']:
                real_column = real_column.to_numpy().reshape(-1, 1)
                synthetic_column = synthetic_data[column_name].to_numpy(
                ).reshape(-1, 1)

                if metadata['fields'][column_name]['type'] in ['categorical']:
                    scores[column_name] = distribution_similarity(
                        real_data=real_column,
                        synthetic_data=synthetic_column,
                        column_names=[column_name],
                        data_type=['categorical'],
                        comparison_type='both',
                        categorical_mapping=True
                    )
                elif metadata['fields'][column_name]['type'] in ['numerical']:
                    scores[column_name] = distribution_similarity(
                        real_data=real_column,
                        synthetic_data=synthetic_column,
                        column_names=[column_name],
                        data_type=['numerical'],
                        comparison_type='both',
                        categorical_mapping=True
                    )

        return scores
