from datetime import datetime
from typing import Tuple, Dict

from src.analysis import Analysis
from src.meta_analysis import MetaAnalysis
from src.dataset import Dataset
from src.metric_enum import MetricEnum
from src.model_enum import ModelEnum
from src.suffix_enum import SuffixEnum


def train(dataset: Dataset, timestamp: str, notes: str) -> Tuple[Dict, Dict]:
    analysis = Analysis(dataset, timestamp, notes)
    svc_results = []
    logistic_regression_results = []
    for key, value in analysis.segments.items():
        for selector in [ModelEnum.ELASTIC_NET, ModelEnum.LOGISTIC_REGRESSION,
                         ModelEnum.SVR]:
            svc_results.append(
                analysis.svc(data=value,
                             segment=str(key),
                             feature_selector=selector))

            logistic_regression_results.append(
                analysis.logistic_regression(data=value,
                                             segment=str(key),
                                             feature_selector=selector))
    svc_result = analysis.analyse_segment('SVC', svc_results)
    lr_result = analysis.analyse_segment('LogisticRegression',
                                         logistic_regression_results)

    return svc_result, lr_result


def main():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

    dataset = Dataset(dataset_name='ltf_03.pkl', calculate_features=True)
    dataset_avg = Dataset(dataset_name='ltf_03_var.pkl',
                          calculate_features=False,
                          suffix=[SuffixEnum.AVG])
    dataset_raw = Dataset(dataset_name='ltf_03_var.pkl',
                          calculate_features=False,
                          suffix=[SuffixEnum.RAW])
    dataset_var = Dataset(dataset_name='ltf_03_var.pkl',
                          calculate_features=False,
                          suffix=[SuffixEnum.VAR])
    dataset_var_avg = Dataset(dataset_name='ltf_03_var.pkl',
                              calculate_features=False,
                              suffix=[SuffixEnum.VAR, SuffixEnum.AVG])

    train(dataset_avg, timestamp, 'avg')
    train(dataset_raw, timestamp, 'raw')
    train(dataset_var, timestamp, 'var')
    train(dataset_var_avg, timestamp, 'var_avg')

    meta_analysis = MetaAnalysis(timestamp)
    meta_analysis.find_best_models()
    meta_analysis.compare_single_segment_results(MetricEnum.PR_AUC)
    meta_analysis.compare_multi_segment_results(MetricEnum.PR_AUC)
    meta_analysis.compare_auc(ModelEnum.SVC, 'var_avg', 'avg', 2508, 165)
    meta_analysis.compare_auc(ModelEnum.LOGISTIC_REGRESSION, 'var_avg', 'avg',
                              2508, 165)
    meta_analysis.compare_auc(ModelEnum.SVC, 'var', 'avg', 2508, 165)
    meta_analysis.compare_auc(ModelEnum.LOGISTIC_REGRESSION, 'var', 'avg',
                              2508, 165)
    meta_analysis.compare_auc(ModelEnum.SVC, 'var_avg', 'raw', 2508, 165)
    meta_analysis.compare_auc(ModelEnum.LOGISTIC_REGRESSION, 'var_avg', 'raw',
                              2508, 165)
    meta_analysis.compare_auc(ModelEnum.SVC, 'raw', 'avg', 2508, 165)
    meta_analysis.compare_auc(ModelEnum.LOGISTIC_REGRESSION, 'raw', 'avg',
                              2508, 165)

    meta_analysis.compare_segments(ModelEnum.SVC, 2508, 165, 16)


if __name__ == '__main__':
    main()
