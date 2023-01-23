"""Timeseries quality report"""
import os
import sys
import pickle
import random
import warnings
import pkg_resources
import dash
import inspect
import importlib
import pprint


from dash import dcc, html
from dash.dependencies import Input, Output
from tqdm import tqdm
from config_io import Config
from collections import OrderedDict


class QualityReport():
    def __init__(self, config_file=None):
        self._config = Config.load_from_file(
            config_file, default="config_quality_report.json",
            default_search_paths=[os.path.dirname(
                inspect.getfile(self.__class__))]
        )

    def _print_scores(self, scores, out):
        for col, score in scores.items():
            assert len(score) >= 1, \
                "At least numerical score has to be generated"

            out.write(f"Column: {col}\n")
            out.write(f"Numeric score: {score[0]}\n")
            # Display figure
            if len(score) == 2:
                score[1].show()

    def visualize(self):
        app = dash.Dash(__name__)
        html_children = []
        for metric, scores in self.dict_metric_scores.items():
            html_children.append(html.Div(
                html.H1(children=metric)))
            for submetric, score_and_plot in scores.items():
                score = score_and_plot[0]
                if len(score_and_plot) > 1:
                    fig = score_and_plot[1]
                    html_children.append(
                        html.Div([
                            html.Div(children=submetric),

                            dcc.Graph(
                                id=f'graph-{metric}-{submetric}',
                                figure=fig,
                                style={'width': '100vh'}
                            )
                        ])
                    )
        app.layout = html.Div(children=html_children)
        app.run_server(debug=True)

    def generate(self, real_data, synthetic_data, metadata, out=sys.stdout):
        self.dict_metric_scores = OrderedDict()

        for metric_type, metrics in self._config["metrics"].items():
            # fidelity/privacy
            metric_module = importlib.import_module(
                f"sdmetrics.timeseries.{metric_type}")
            self.dict_metric_scores[metric_type] = OrderedDict()
            for metric_dict in metrics:
                metric_name = list(metric_dict.keys())[0]
                metric_config = list(metric_dict.values())[0]
                metric_class = getattr(metric_module, metric_config["class"])

                # Metrics that do not have `target` (e.g., session length)
                if "target_list" not in metric_config:
                    self.dict_metric_scores[metric_type][metric_name] = metric_class.compute(
                        real_data, synthetic_data, metadata)

                # Metrics that have `target` (e.g., single attribute distributional similarity)
                else:
                    self.dict_metric_scores[metric_type][metric_name] = \
                        OrderedDict()
                    for target in metric_config["target_list"]:
                        self.dict_metric_scores[metric_type][metric_name][
                            str(target)] = metric_class.compute(
                            real_data, synthetic_data, metadata, target=target)

        pprint.pprint(self.dict_metric_scores)

        # for metric in tqdm(METRICS):
        #     try:
        #         self.dict_metric_scores[metric.name] = metric.compute(
        #             real_data, synthetic_data, metadata)
        #     except:
        #         attribute_cols = metadata['entity_columns'] + metadata['context_columns']
        #         feature_cols = list(set(real_data.columns) -
        #                             set(attribute_cols))
        #         if metric == CrossFeatureCorrelation:
        #             self.dict_metric_scores[metric.name] = metric.compute(
        #                 real_data, synthetic_data, metadata,
        #                 target=random.choices(
        #                     [f for f in feature_cols
        #                      if metadata['fields'][f]['type']
        #                      == 'numerical'],
        #                     k=2))

        #         elif metric == PerFeatureAutocorrelation:
        #             self.dict_metric_scores[metric.name] = metric.compute(
        #                 real_data, synthetic_data, metadata,
        #                 target=random.choice(
        #                     [f for f in feature_cols
        #                      if metadata['fields'][f]['type']
        #                      == 'numerical']))

        #         elif metric == SingleAttrSingleFeatureCorrelation:
        #             self.dict_metric_scores[metric.name] = metric.compute(
        #                 real_data, synthetic_data, metadata,
        #                 attr_name=random.choice(
        #                     [f for f in attribute_cols
        #                      if metadata['fields'][f]['type']
        #                      == 'categorical']),
        #                 feature_name=random.choice(
        #                     [f for f in feature_cols
        #                      if metadata['fields'][f]['type']
        #                      == 'numerical'])
        #             )

        #         else:
        #             self.dict_metric_scores[metric.name] = None

    def save(self, filepath):
        """Save this report instance to the given path using pickle.

        Args:
            filepath (str):
                The path to the file where the report instance will be serialized.
        """
        self._package_version = pkg_resources.get_distribution(
            'sdmetrics').version

        with open(filepath, 'wb') as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, filepath):
        """Load a ``QualityReport`` instance from a given path.

        Args:
            filepath (str):
                The path to the file where the report is stored.

        Returns:
            QualityReort:
                The loaded quality report instance.
        """
        current_version = pkg_resources.get_distribution('sdmetrics').version

        with open(filepath, 'rb') as f:
            report = pickle.load(f)
            if current_version != report._package_version:
                warnings.warn(
                    f'The report was created using SDMetrics version `{report._package_version}` '
                    f'but you are currently using version `{current_version}`. '
                    'Some features may not work as intended.')

            return report
