"""Timeseries quality report"""
from sdmetrics.timeseries import TimeSeriesMetric


class QualityReport():
    def generate(self, real_data, synthetic_data, metadata):
        print(TimeSeriesMetric.get_subclasses())
