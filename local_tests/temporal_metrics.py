from sdmetrics.demos import load_timeseries_demo
from sdmetrics.timeseries import SingleAttrDistSimilarity

real_data, synthetic_data, metadata = load_timeseries_demo()
print(metadata)
# print(metadata["entity_columns"], metadata["context_columns"])
print(real_data.head())

scores = SingleAttrDistSimilarity.compute(real_data, synthetic_data, metadata)
print(scores)
