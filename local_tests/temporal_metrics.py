import matplotlib.pyplot as plt

from sdmetrics.demos import load_timeseries_demo
from sdmetrics.timeseries import SingleAttrDistSimilarity, SingleAttrCoverage

real_data, synthetic_data, metadata = load_timeseries_demo()
print(metadata)
# print(metadata["entity_columns"], metadata["context_columns"])
print(real_data.head())

# scores = SingleAttrDistSimilarity.compute(real_data, synthetic_data, metadata)
scores = SingleAttrCoverage.compute(real_data, synthetic_data, metadata)

for col, score in scores.items():
    print(score)
    print(f"Column: {col}")
    for score_ in score:
        try:
            plt.show()
        except:
            print(score_)
