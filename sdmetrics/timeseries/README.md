# Time Series Metrics

## Installation
Create and enter the anaconda virtual environment
```Bash
conda create --name <your_env_name> python=3.9
conda activate <your_env_name>
```

Install packages and dependencies
```
pip3 install -e .
```

## Implemented metrics

A list of implemented metrics:

- [x] [Single attribute distributional similarity](./fidelity/single_attr_dist_similarity.py)
- [x] [Single attirubte coverage](./fidelity/single_attr_coverage.py)
- [x] [Number of records per object](./fidelity/session_length_dist_similarity.py)
- [x] [Feature Distribution Similarity](./fidelity/feature_dist_similarity.py)
- [x] [Cross-Feature Correlations](./fidelity/cross_feature_correlation.py)
- [x] [Per-feature autocorrelation](./fidelity/perfeature_autocorrelation.py)
- [x] [Inter-arrival time](./fidelity/interarrival_dist_similarity.py)
- [x] [Qualitative metadata-feature correlation](./fidelity/single_attr_single_feature_correlation.py)

All the timeseries metrics are subclasses form the `sdmetrics.timeseries.TimeSeriesMetric`
class, which can be used to locate all of them:

```Python
from sdmetrics.timeseries import TimeSeriesMetric
TimeSeriesMetric.get_subclasses()
```

## Usage
### Report (a set of metrics)
Get started with **SDMetrics Reports** using some demo data,

```Python
from sdmetrics.demos import load_timeseries_demo
from sdmetrics.reports.timeseries import QualityReport

real_data, synthetic_data, metadata = load_timeseries_demo()

my_report = QualityReport()
my_report.generate(real_data, synthetic_data, metadata)
```
which will generate a report containing numerical values and visual plots.

### Single metric
**Want more metrics?** You can also manually apply any of the metrics in this library to your data. All the timeseries metrics operate on *at least* three inputs:

* `real_data`: A `pandas.DataFrame` with the data from the real dataset.
* `synthetic_data`: A `pandas.DataFrame` with the data from the synthetic dataset.
* `entity_columns`: A `list` indicating which columns represent entities to which
  the different senquences from the dataset belong.

For example, an `SingleAttrDistSimilarity` metric can be used on the `sunglasses` demo data as follows:

```python3
from sdmetrics.demos import load_timeseries_demo
from sdmetrics.timeseries import SingleAttrDistSimilarity

real_data, synthetic_data, metadata = load_timeseries_demo()
scores = SingleAttrDistSimilarity.compute(real_data, synthetic_data, metadata)
```

The output `scores` will contain two parts (if applicable) for each attribute/feature: (1) a numerical number (2) a visualization plot.

<img src="../../docs/images/timeseries_sunglass_region_distribution.png" width="480">

### Single dataset
For a quick check on real (or synthetic) dataset alone, you can use `SingleDatasetVisualize`:

```Python
from sdmetrics.demos import load_timeseries_demo
from sdmetrics.reports.timeseries import SingleDatasetVisualize

real_data, synthetic_data, metadata = load_timeseries_demo()

my_report = SingleDatasetVisualize()
my_report.visualize(real_data, metadata)
```
which will generate a set of plots to visualize the distribution of different fields.

---

Additionally, all the metrics accept a `metadata` argument which must be a dict following
the [Metadata JSON schema from SDV](https://docs.sdv.dev/sdmetrics/getting-started/metadata/sequential-metadata), which will be used to determine which columns are compatible
with each one of the different metrics, as well as to extract any additional information required
by the metrics, such as the `entity_columns`.

If this dictionary is not passed it will be built based on the data found in the real table,
but in this case some field types may not represent the data accurately (e.g. categorical
columns that contain only integer values will be seen as numerical), and any additional
information required by the metrics will not be populated.


# TODOs:
- [ ] add README on adding new metrics
- [ ] save/load reports including figures and numbers
- [ ] single dataset visualization
- [ ] Convert matplotlib to plotly
- [ ] Modify metadata def from SDV?
- [ ] Better plot information
- [ ] More informative reports
- [ ] Create plot examples for pcap/netflow/wiki