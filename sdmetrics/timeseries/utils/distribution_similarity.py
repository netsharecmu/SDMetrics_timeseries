import warnings
import numpy as np
import torch
from geomloss import SamplesLoss
from typing import Literal, Optional, Dict, List
from collections import Counter, OrderedDict
from .distance import jsd, emd


def distribution_similarity(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    data_type: List[Literal['categorical', 'numeric']],
    comparison_type: Literal['quantitative', 'qualitative', 'both'],
    categorical_mapping: Optional[bool] = None
):
    """Computes the quantitative and/or qualitative similarity between two distributions

    Inputs:
        Synthetic data (represented as a data array)
        Real data (represented as a data array)
        Data type (categorical, continuous)
        Type of comparison (quantitative, qualitative, both)
        Categorical mapping: search exact mapping of values
    """
    assert len(real_data.shape) == len(synthetic_data.shape) == 2, \
        "Both real data and synthetic data must be 2D array. " \
        "For 1D array, use reshape(-1, 1) for conversion."

    assert real_data.shape[1] == synthetic_data.shape[1] == len(data_type), \
        "Real data and synthetic data must have the same dimension. " \
        "Each dimension of data has to be speicified as `categorical` or `numeric`."

    output = []

    # Quantitative
    if comparison_type in ['quantitative', 'both']:
        # categorical only
        if set(data_type) == {'categorical'}:
            assert categorical_mapping is not None, \
                "Categorical variable, `categorical_mapping` must be set."
            output.append(jsd(real_data, synthetic_data, categorical_mapping))

        # numeric only
        elif set(data_type) == {'numeric'}:
            output.append(emd(real_data, synthetic_data))

        # categorical/numeric mixed: discretizing numeric variables
        # TODO: think of alternative strategy
        elif set(data_type) == {'categorical', 'numeric'}:
            pass
        else:
            raise ValueError(
                "Unsupported data type, only `categorical` and `numeric` are supported.")

    # Qualitative
    if comparison_type in ['qualitative', 'both']:
        # 1D array
        if real_data.shape[1] == 1:

    return output
