import os
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from functools import reduce
import operator
import seaborn as sns
from itertools import combinations
from src.metric_enum import MetricEnum
from src.model_enum import ModelEnum
from statsmodels.stats.multitest import multipletests

plt.style.use('ggplot')


class MetaAnalysis:

    def __init__(self, timsetamp: str):
        self.timestamp = timsetamp
        self.dataset = pd.read_csv(
            f'output/{self.timestamp}/segment_report.csv', index_col=0)

        self.reports = {
            'raw': None,
            'var': None,
            'avg': None,
            'var_avg': None
        }

        self.root_directory = f"output/{self.timestamp}/meta-analysis"
        if not os.path.exists(self.root_directory):
            os.makedirs(self.root_directory)

    @staticmethod
    def _is_better(df1: pd.Series, df2: pd.Series) -> bool:
        """
        Compare metrics between two series.

        Parameters
        ----------
        df1: pandas.DataFrame
        df2 pandas.DataFrame

        Returns
        -------
        Bool
        """
        if df1['pr_auc'] > df2['pr_auc']:
            return False
        elif df1['pr_auc'] < df2['pr_auc']:
            return True
        elif df1['pr_auc'] == df2['pr_auc']:
            if df1['roc_auc'] > df2['roc_auc']:
                return False
            elif df1['roc_auc'] < df2['roc_auc']:
                return True
            elif df1['roc_auc'] == df2['roc_auc']:
                if df1['precision'] > df2['precision']:
                    return False
                elif df1['precision'] < df2['precision']:
                    return True
                elif df1['precision'] == df2['precision']:
                    if df1['accuracy'] > df2['accuracy']:
                        return False
                    else:
                        return True

    def _get_best_models(self, segment_report: pd.DataFrame) -> Dict:
        """
        Generate dictionary with best models.

        Parameters
        ----------
        segment_report: pandas.DataFrame

        Returns
        -------
        Dict
            Dictionary with best models for each segment
        """
        best = {}
        for segment, segment_data in segment_report.groupby(
                segment_report['segment']):
            best[segment] = {}
            for classifier, classifier_data in segment_data.groupby(
                    segment_data['classifier']):
                for key, row in classifier_data.iterrows():
                    if classifier not in best[segment].keys():
                        best[segment][classifier] = row
                        continue
                    if self._is_better(best[segment][classifier], row):
                        best[segment][classifier] = row

        return best

    @staticmethod
    def _get_variant_result(best: Dict) -> Dict:
        """
        Generate dictionary with results.

        Parameters
        ----------
        best: Dict

        Returns
        -------
        Dict
        """
        results = {
            'segments': [],
            'svc': {
                MetricEnum.PR_AUC.value: [],
                MetricEnum.ROC_AUC.value: [],
                MetricEnum.PRECISION.value: [],
                MetricEnum.ACCURACY.value: [],
            },
            'logistic_regression': {
                MetricEnum.PR_AUC.value: [],
                MetricEnum.ROC_AUC.value: [],
                MetricEnum.PRECISION.value: [],
                MetricEnum.ACCURACY.value: [],
            },
        }
        for segment, data in best.items():
            if 'best' not in results.keys():
                results['best'] = {}
            results['segments'].append(segment)
            results['svc'][MetricEnum.PR_AUC.value].append(
                data['SVC'][MetricEnum.PR_AUC.value])
            results['logistic_regression'][MetricEnum.PR_AUC.value].append(
                data['LogisticRegression'][MetricEnum.PR_AUC.value])
            results['svc'][MetricEnum.ROC_AUC.value].append(
                data['SVC'][MetricEnum.ROC_AUC.value])
            results['logistic_regression'][MetricEnum.ROC_AUC.value].append(
                data['LogisticRegression'][MetricEnum.ROC_AUC.value])
            results['svc'][MetricEnum.PRECISION.value].append(
                data['SVC'][MetricEnum.PRECISION.value])
            results['logistic_regression'][MetricEnum.PRECISION.value].append(
                data['LogisticRegression'][MetricEnum.PRECISION.value])
            results['svc'][MetricEnum.ACCURACY.value].append(
                data['SVC'][MetricEnum.ACCURACY.value])
            results['logistic_regression'][MetricEnum.ACCURACY.value].append(
                data['LogisticRegression'][MetricEnum.ACCURACY.value])
            results['best'][segment] = best[segment]
        return results

    def find_best_models(self):
        for variant in ['raw', 'var', 'avg', 'var_avg']:
            best = self._get_best_models(
                self.dataset.loc[self.dataset['notes'] == variant])
            self.reports[variant] = self._get_variant_result(best)

        return self.reports

    def compare_single_segment_results(self, metric: MetricEnum):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        width = 0.2

        var = reduce(operator.concat,
                     [self.reports['var']['svc'][metric.value],
                      self.reports['var']['logistic_regression'][
                          metric.value]])
        avg = reduce(operator.concat,
                     [self.reports['avg']['svc'][metric.value],
                      self.reports['avg']['logistic_regression'][
                          metric.value]])
        var_avg = reduce(operator.concat, [
            self.reports['var_avg']['svc'][metric.value],
            self.reports['var_avg']['logistic_regression'][metric.value]])
        labels = ['SVC', 'LogisticRegression']
        x = np.arange(len(labels))

        var_bar = ax.bar(x - width, var, width, label='VAR')
        avg_bar = ax.bar(x, avg, width, label='AVG')
        var_avg_bar = ax.bar(x + width, var_avg, width, label='VAR_AVG')

        ax.set_ylabel('Scores')
        ax.set_title(f'{metric.value.upper()} Scores')
        ax.set_xticks(x, labels)
        ax.legend(loc='lower center')

        ax.bar_label(var_bar, padding=3, fmt='%.4f', label_type='center',
                     rotation=90)
        ax.bar_label(avg_bar, padding=3, fmt='%.4f', label_type='center',
                     rotation=90)
        ax.bar_label(var_avg_bar, padding=3, fmt='%.4f', label_type='center',
                     rotation=90)

        plt.tight_layout()
        plt.savefig(f'{self.root_directory}/single_{metric.value}_compare.png')

    def compare_multi_segment_results(self, metric: MetricEnum,
                                      variant: str = 'var_avg'):
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        width = 0.24

        number_of_segments = len(self.reports['raw']['segments'])
        single_svc = self.reports[variant]['svc'][
                         metric.value] * number_of_segments
        single_lr = self.reports[variant]['logistic_regression'][
                        metric.value] * number_of_segments
        raw_svc = self.reports['raw']['svc'][metric.value]
        raw_lr = self.reports['raw']['logistic_regression'][metric.value]

        x = np.arange(len(self.reports['raw']['segments']))

        raw_bar_svc = ax.bar(x - width * 1.5, raw_svc, width, label='SVC raw')
        var_bar_svc = ax.bar(x - width / 2, single_svc, width,
                             label='SVC variant')
        ax.bar_label(raw_bar_svc, padding=3, fmt='%.4f', label_type='center',
                     rotation=90)
        ax.bar_label(var_bar_svc, padding=3, fmt='%.4f', label_type='center',
                     rotation=90)

        raw_bar_lr = ax.bar(x + width / 2, raw_lr, width,
                            label='Logistic Regression raw')
        var_bar_lr = ax.bar(x + width * 1.5, single_lr, width,
                            label='Logistic Regression variant')
        ax.bar_label(raw_bar_lr, padding=3, fmt='%.4f', label_type='center',
                     rotation=90)
        ax.bar_label(var_bar_lr, padding=3, fmt='%.4f', label_type='center',
                     rotation=90)

        ax.set_ylabel('Scores')
        ax.set_xticks(x, self.reports['raw']['segments'])
        ax.legend(loc='lower center')

        fig.suptitle(f"{metric.value.upper()} Scores", fontsize=24)
        plt.tight_layout()
        plt.savefig(
            f'{self.root_directory}/multiple_{metric.value}_compare.png')

    @staticmethod
    def _calculate_significance_of_difference(
            neg_cases_a: float = 0.0,
            neg_cases_b: float = 0.0,
            pos_cases_a: float = 0.0,
            pos_cases_b: float = 0.0,
            auc_a: float = 0.0,
            auc_b: float = 0.0,
            correction: int = 16
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        Significance of the Difference between the Areas under Two Independent
        Curves.

        The algorithm implements an approximation formula P(z) of the Normal
        Distribution between the limits −z to z.

        P(z) = 1 - (1/(1 + a1z + a2z^2 + a3z^3 + a4z^4 + a5z^5 + a6z^6)^16)

        The coefficients a1 to a6 are:
        a1=0.04986734697
        a2=0.02114100615
        a3=0.00327762632
        a4=0.000038603575
        a5=0.000048896635
        a6=0.000005382975

        Parameters
        ----------
        neg_cases_a: float
            number of actually negative cases sample A
        neg_cases_b: float
            number of actually negative cases sample B
        pos_cases_a: float
            number of actually positive cases sample A
        pos_cases_b: float
            number of actually positive cases sample B
        auc_a: float
            area under curve sample A
        auc_b: float
            area under curve sample B
        correction: int
            Bonferroni correction

        Returns
        -------
        Tuple
            standard error A, standard error B, difference: areaA—areaB,
            standard error of the difference, Z,
            P-value: non-directional (two-tailed),
            P-value: directional (one-tailed)
        """
        quartile_1a = auc_a / (2 - auc_a)
        quartile_1b = auc_b / (2 - auc_b)

        quartile_2a = (2 * pow(auc_a, 2)) / (1 + auc_a)
        quartile_2b = (2 * pow(auc_b, 2)) / (1 + auc_b)

        std_error_a = ((auc_a * (1 - auc_a))
                       + ((pos_cases_a - 1) * (quartile_1a - pow(auc_a, 2)))
                       + ((neg_cases_a - 1) * (quartile_2a - pow(auc_a, 2))))
        std_error_a = round(sqrt(std_error_a / (pos_cases_a * neg_cases_a)), 4)
        std_error_b = ((auc_b * (1 - auc_b))
                       + ((pos_cases_b - 1) * (quartile_1b - pow(auc_b, 2)))
                       + ((neg_cases_b - 1) * (quartile_2b - pow(auc_b, 2))))
        std_error_b = round(sqrt(std_error_b / (pos_cases_b * neg_cases_b)), 4)
        std_error_diff = round(sqrt(pow(std_error_a, 2) + pow(std_error_b, 2)),
                               4)

        z = abs(round((auc_a - auc_b) / std_error_diff, 4))
        a1 = 0.04986734697
        a2 = 0.02114100615
        a3 = 0.00327762632
        a4 = 0.000038603575
        a5 = 0.000048896635
        a6 = 0.000005382975
        p2 = round(
            pow(
                (((((a6 * z + a5) * z + a4) * z + a3) * z + a2) * z + a1)
                * z + 1, -16
            ), 6) * correction
        p1 = round(p2 / 2, 6)

        return (std_error_a, std_error_b, round(auc_a - auc_b, 6),
                std_error_diff, z, p1, p2)

    def _compare(self, row: pd.Series, negative_cases: int,
                 positive_cases: int,
                 correction: int) -> pd.Series:
        """
        Compare AUC.

        Negative cases: 2354
        Positive cases: 164
        
        Parameters
        ----------
        row: pd.Series
        negative_cases: int
        positive_cases: int
        correction: int
            Bonferroni correction

        Returns
        -------
        pandas.Series
        """
        compare = self._calculate_significance_of_difference(negative_cases,
                                                             negative_cases,
                                                             positive_cases,
                                                             positive_cases,
                                                             row['AUC A'],
                                                             row['AUC B'],
                                                             correction)

        return pd.concat(
            [row, pd.Series(compare,
                            index=['standard error A',
                                   'standard error B',
                                   'difference: areaA—areaB',
                                   'standard error of the difference',
                                   'Z',
                                   'P-value: non-directional (two-tailed)',
                                   'P-value: directional (one-tailed)'])
             ], sort=False)

    def compare_auc(self, model: ModelEnum, variant1: str, variant2: str,
                    negative_cases: int,
                    positive_cases: int, correction: int = 1) -> pd.DataFrame:
        """
        Compare AUC of two variants.

        Parameters
        ----------
        model: ModelEnum
        variant1: str
        variant2: str
        negative_cases: int
        positive_cases: int
        correction: int
            Bonferroni correction

        Returns
        -------
        pandas.DataFrame
        """
        variant1_values = self.reports[variant1][model.value]['pr_auc']
        variant2_values = self.reports[variant2][model.value]['pr_auc']
        segments = self.reports[variant1]['segments']

        if variant1 == 'raw':
            variant2_values = self.reports[variant2][model.value][
                                  'pr_auc'] * len(variant1_values)
        elif variant2 == 'raw':
            segments = self.reports[variant2]['segments']
            variant1_values = self.reports[variant1][model.value][
                                  'pr_auc'] * len(variant2_values)

        df = pd.DataFrame(data=zip(segments, variant1_values, variant2_values),
                          index=np.arange(len(segments)),
                          columns=['Segment', 'AUC A', 'AUC B'])
        df = df.apply(
            lambda x: self._compare(x, negative_cases, positive_cases,
                                    correction),
            axis=1).reindex(
            columns=['Segment', 'AUC A', 'AUC B', 'standard error A',
                     'standard error B',
                     'difference: areaA—areaB',
                     'standard error of the difference', 'Z',
                     'P-value: non-directional (two-tailed)',
                     'P-value: directional (one-tailed)'])

        df.to_csv(
            f'{self.root_directory}/compare_{model.value}_auc_{variant1}_'
            f'vs_{variant2}.csv')

        return df

    def compare_segments(self, model: ModelEnum, negative_cases: int,
                         positive_cases: int,
                         correction: int = 1) -> pd.DataFrame:
        variant1_values = self.reports['raw'][model.value]['pr_auc']
        variant2_values = self.reports['raw'][model.value]['pr_auc']
        segments = self.reports['raw']['segments']

        df_list = []
        for combination in combinations(segments, 2):
            df = pd.DataFrame(data=zip([combination[0]],
                                       [combination[1]],
                                       [variant1_values[
                                            segments.index(combination[0])]],
                                       [variant2_values[
                                            segments.index(combination[1])]]),
                              columns=['Segment A', 'Segment B', 'AUC A',
                                       'AUC B'])
            df_list.append(
                df.apply(
                    lambda x: self._compare(x, negative_cases, positive_cases,
                                            1),
                    axis=1).reindex(
                    columns=['Segment A', 'Segment B', 'AUC A', 'AUC B',
                             'standard error A', 'standard error B',
                             'difference: areaA—areaB',
                             'standard error of the difference', 'Z',
                             'P-value: non-directional (two-tailed)',
                             'P-value: directional (one-tailed)']))
        result = pd.concat(df_list, axis=0)
        result.to_csv(f'{self.root_directory}/compare_raw_segments.csv')
        result_corrected = multipletests(
            result['P-value: directional (one-tailed)'],
            method='bonferroni')
        result['P-value: directional (one-tailed)'] = result_corrected[1]
        data = result.groupby(
            ['Segment A', 'Segment B']
        )['P-value: directional (one-tailed)'].median()
        data = data.unstack(level=0)

        sns.set_theme()
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        sns.heatmap(data, vmin=0, vmax=0.5, annot=True, fmt=".3f",
                    cmap="YlGnBu_r")
        fig.suptitle(f"Single segments significance comparison", fontsize=24)
        plt.tight_layout()
        plt.savefig(f'{self.root_directory}/heatmap_{correction}.png')

        return df
