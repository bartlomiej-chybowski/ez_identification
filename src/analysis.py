import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

from src.dataset import Dataset
from sklearn.svm import SVR, SVC
from src.model_enum import ModelEnum
from typing import Dict, Tuple, List
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import roc_curve, auc, precision_recall_curve, \
    PrecisionRecallDisplay, RocCurveDisplay, accuracy_score, precision_score


class Analysis:

    def __init__(self, dataset: Dataset, timestamp: str, notes: str = ''):
        self.dataset = dataset
        self.data = dataset.dataset
        self.patients_ids = self.data.loc[:, 'patient_id'].unique()
        self.segments = self._get_segments()
        (self.hfo_features, self.bivariate_features,
         self.univariate_features) = self._get_features()
        self.notes = notes
        self.timestamp = timestamp
        self.root_directory = f"output/{self.timestamp}"
        if not os.path.exists(self.root_directory):
            os.makedirs(self.root_directory)

    def _get_segments(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        segments = {}
        self.data.loc[:, 'pathology'] = \
            self.data.loc[:, 'pathology'].astype('category').cat.codes
        self.data.insert(4, 'target',
                         self.data['pathology'] & self.data['resected'])

        for segment, content in self.data.groupby(self.data['segm']):
            segments[segment] = {
                'x': content.loc[:, ~content.columns.isin(
                    ['outcome', 'pathology', 'segm', 'segm_type',
                     'segm_number', 'channel_name', 'resected', 'target'])],
                'y': content.loc[:, ['patient_id', 'channel_name', 'pathology',
                                     'resected', 'target']],
            }

        return segments

    def _get_features(self) -> Tuple[List[str], List[str], List[str]]:
        features = self.segments['1_0']['x'].columns

        hfo_stumps = ['MNI', 'spike', 'HFO']
        hfo_features = [x for x in features if
                        any(x.startswith(s) for s in hfo_stumps)]

        bivariate_stumps = ['xcorr', 'ren', 'pli', 'phase_sync', 'phase_const',
                            'lin_corr', 'coherence']
        bivariate_features = [x for x in features if
                              any(x.startswith(s) for s in bivariate_stumps)]

        uni_stumps = ['power', 'fac', 'pac', 'pse', 'hlx', 'lfr']
        univariate_features = [x for x in features if
                               any(x.startswith(s) for s in uni_stumps)]

        return hfo_features, bivariate_features, univariate_features

    @staticmethod
    def _scale(x_train: pd.DataFrame) -> Tuple[np.array, StandardScaler]:
        scaler = StandardScaler()
        transformed_data = scaler.fit_transform(x_train)

        return transformed_data, scaler

    def _fit(self, classifier: RandomizedSearchCV, data: Dict,
             final_features: List, patient: int,
             segment: str, feature_selector: ModelEnum,
             output: mp.Queue) -> Dict:

        x_train = data['x'].loc[~data['x']['patient_id'].isin([patient])]
        y_train = data['y'].loc[~data['y']['patient_id'].isin([patient])]
        x_test = data['x'].loc[data['x']['patient_id'].isin([patient])]
        y_test = data['y'].loc[data['y']['patient_id'].isin([patient])]

        x_train, scaler = self._scale(
            x_train.loc[:, x_train.columns.isin(final_features)])

        classifier = classifier.best_estimator_
        classifier = classifier.fit(x_train, y_train.loc[:, 'target'])

        y_pred = classifier.predict(
            scaler.transform(
                x_test.loc[:, x_test.columns.isin(final_features)]))

        if getattr(classifier, "decision_function", False):
            y_score = classifier.decision_function(
                scaler.transform(
                    x_test.loc[:, x_test.columns.isin(final_features)]))
        else:
            y_score = classifier.predict_proba(
                scaler.transform(
                    x_test.loc[:, x_test.columns.isin(final_features)]))[:, 1]

        accuracy = accuracy_score(y_test.loc[:, 'target'], y_pred)
        precision = precision_score(y_test.loc[:, 'target'], y_pred,
                                    average='weighted', zero_division=0)
        roc_auc, pr_auc = self._plot_fit_outcome(accuracy, classifier, patient,
                                                 precision, segment,
                                                 y_pred, y_score, y_test,
                                                 feature_selector)
        result = {
            'patient_id': patient,
            'segment': segment,
            'accuracy': accuracy,
            'precision': precision,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'predictions': y_pred
        }
        output.put(result)

        return result

    def _plot_fit_outcome(self, accuracy: float,
                          classifier: RandomizedSearchCV, patient: int,
                          precision: float, segment: str, y_pred, y_score,
                          y_test,
                          feature_selector: ModelEnum) -> Tuple:
        fig, ax = plt.subplots(3, 1, figsize=(15, 8))
        ax = ax.ravel()

        try:
            fpr, tpr, _ = roc_curve(y_test.loc[:, 'target'].ravel(),
                                    y_score.ravel())
            precision_pr, recall, _ = precision_recall_curve(
                y_test.loc[:, 'target'].ravel(),
                y_score.ravel())
        except ValueError:
            fpr, tpr, _ = roc_curve(y_test.loc[:, 'target'].ravel(),
                                    y_pred.ravel())
            precision_pr, recall, _ = precision_recall_curve(
                y_test.loc[:, 'target'].ravel(),
                y_pred.ravel())
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision_pr)
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot(linestyle='--', marker='o',
                                               ax=ax[0])
        PrecisionRecallDisplay(precision=precision_pr, recall=recall).plot(
            linestyle='--', marker='o', ax=ax[1])
        y_test = y_test.reset_index(drop=True)
        y_channel = y_test.loc[:, 'channel_name']
        ax[2].scatter(y_channel, y_test.loc[y_channel.index, 'target'],
                      color='red', marker='o')
        ax[2].scatter(y_channel, y_test.loc[y_channel.index, 'pathology'],
                      color='yellow', marker='+')
        ax[2].scatter(y_channel, y_pred[list(y_channel.index)], color='blue',
                      marker='x')
        ax[2].tick_params(labelrotation=90)
        ax[2].set_title(f"Patient: {patient}")
        fig.suptitle(
            f"Segment: {segment}, accuracy: {accuracy}, "
            f"precision: {precision}, ROC_AUC: {roc_auc:.3f}, "
            f"PR_AUC: {pr_auc:.3f}")
        plt.subplots_adjust(hspace=1)
        class_name = str(classifier.__class__.__name__)

        try:
            if not os.path.exists(
                    f'{self.root_directory}/{class_name}/{self.notes}'
                    f'/{feature_selector.value}/{segment}'):
                os.makedirs(f'{self.root_directory}/{class_name}/{self.notes}'
                            f'/{feature_selector.value}/{segment}')
        except FileExistsError:
            pass

        plt.savefig(
            f'{self.root_directory}/{class_name}/{self.notes}/'
            f'{feature_selector.value}/{segment}/{patient}.png')
        plt.close()

        return roc_auc, pr_auc

    def _cross_validate(self, classifier: RandomizedSearchCV,
                        final_features: List, data: Dict,
                        segment: str, feature_selector: ModelEnum) -> List:
        """
        Run cross-validation simultaneously.

        Parameters
        ----------
        classifier: RandomizedSearchCV
        final_features: List
        data: Dict
        segment: str
        feature_selector: ModelEnum

        Returns
        -------
        List
        """
        output = mp.Queue()
        processes = []

        for patient in self.patients_ids:
            process = mp.Process(target=self._fit,
                                 args=(classifier, data, final_features,
                                       patient, segment, feature_selector,
                                       output,))
            process.start()
            processes.append(process)

        for p in processes:
            p.join()

        return [output.get() for _ in processes]

    def select_features(self, data: Dict, feature_selector: ModelEnum):
        """
        Select the most important features.

        Parameters
        ----------
        data: Dict
        feature_selector: ModelEnum

        Returns
        -------
        Tuple
            List of selected features and Feature Selector
        """
        selectors = []
        final_features = []
        for feature_group in [self.hfo_features, self.bivariate_features,
                              self.univariate_features]:
            x, _ = self._scale(
                data['x'].loc[:, data['x'].columns.isin(feature_group)])
            if feature_selector == ModelEnum.ELASTIC_NET:
                elastic = RandomizedSearchCV(
                    ElasticNet(random_state=42), param_distributions={
                        'alpha': np.around(np.logspace(0.01, 3, 30) / 100,
                                           decimals=3),
                        'l1_ratio': np.linspace(0.1, 1.0, 9, False)
                    },
                    cv=5,
                    n_jobs=-1,
                    random_state=42
                ).fit(x, data['y'].loc[:, 'target'])
                selector = RFECV(elastic.best_estimator_, step=1, cv=3,
                                 scoring='average_precision')
            if feature_selector == ModelEnum.LOGISTIC_REGRESSION:
                logistic_regression = RandomizedSearchCV(
                    LogisticRegression(max_iter=8000, random_state=42),
                    param_distributions={
                        'C': np.around(np.logspace(0.01, 3, 30) / 100,
                                       decimals=3),
                        'penalty': ['l1', 'l2', 'none'],
                        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga']
                    },
                    cv=5,
                    n_jobs=-1,
                    random_state=42
                ).fit(x, data['y'].loc[:, 'target'])
                selector = RFECV(logistic_regression.best_estimator_, step=1,
                                 cv=3, scoring='average_precision')
            if feature_selector == ModelEnum.SVR:
                selector = RFECV(SVR(kernel="linear"), step=1, cv=3,
                                 scoring='average_precision')
            selector = selector.fit(x, data['y'].loc[:, 'target'])
            selectors.append(selector)

            final_features.extend(
                list(pd.DataFrame(data=[feature_group, selector.ranking_]).
                     T[selector.support_].sort_values(by=1)[0].values))

        return final_features, selectors

    def svc(self, data: Dict, segment: str,
            feature_selector: ModelEnum) -> Dict:
        """
        Tune and train with cross-validation Support Vector classifier model.

        Parameters
        ----------
        data: Dict
        segment: str
        feature_selector: ModelEnum

        Returns
        -------
        Dict
        """
        final_features, selectors = self.select_features(data,
                                                         feature_selector)

        x = data['x'].loc[:, data['x'].columns.isin(final_features)]
        x_train, _ = self._scale(x)

        svc_classifier = RandomizedSearchCV(
            SVC(class_weight='balanced', cache_size=4096, random_state=42),
            param_distributions={
                'C': np.around(np.logspace(0.01, 3, 30) / 100, decimals=3),
                'kernel': ['linear', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto'],
                'decision_function_shape': ['ovo', 'ovr']
            },
            cv=3,
            scoring='average_precision',
            n_jobs=-1,
            random_state=42
        ).fit(x_train, data['y'].loc[:, 'target'])

        scores = self._cross_validate(svc_classifier, final_features, data,
                                      segment,
                                      feature_selector)

        return {
            'result': {
                'model': svc_classifier.best_params_,
                'selectors': selectors,
                'final_features': final_features,
                'scores': scores,
                'segment': segment
            }
        }

    def logistic_regression(self, data: Dict, segment: str,
                            feature_selector: ModelEnum) -> Dict:
        """
        Tune and train with cross-validation Logistic Regression model.

        Parameters
        ----------
        data: Dict
        segment: str
        feature_selector: ModelEnum

        Returns
        -------
        Dict
        """
        final_features, selectors = self.select_features(data,
                                                         feature_selector)

        x = data['x'].loc[:, data['x'].columns.isin(final_features)]
        x_train, _ = self._scale(x)

        lr_classifier = RandomizedSearchCV(
            LogisticRegression(class_weight='balanced', max_iter=8000),
            param_distributions={
                'C': np.around(np.logspace(0.01, 3, 30) / 100, decimals=3),
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'solver': ['newton-cg', 'liblinear', 'lbfgs', 'saga']
            },
            cv=3,
            scoring='average_precision',
            n_jobs=-1,
            random_state=42
        ).fit(x_train, data['y'].loc[:, 'target'])

        scores = self._cross_validate(lr_classifier, final_features, data,
                                      segment, feature_selector)

        return {
            'result': {
                'model': lr_classifier.best_params_,
                'selectors': selectors,
                'final_features': final_features,
                'scores': scores,
                'segment': segment,
                'feature_selection': selectors
            }
        }

    def analyse_segment(self, classifier: str, results: List) -> Dict:
        """
        Analyse results in segment and save as csv

        Parameters
        ----------
        classifier: str
        results: LIst

        Returns
        -------
        Dict
        """
        if not os.path.exists(f'{self.root_directory}/segment_report.csv'):
            pd.DataFrame(columns=['segment', 'accuracy', 'precision',
                                  'roc_auc', 'pr_auc', 'classifier',
                                  'best_estimator', 'features',
                                  'feature_selection', 'timestamp', 'dataset',
                                  'notes']
                         ).to_csv(f'{self.root_directory}/segment_report.csv')

        segment_result_report = pd.read_csv(
            f'{self.root_directory}/segment_report.csv', index_col=0)
        patients_results = {}
        segment_results = []
        for index, result in enumerate(results):
            accuracy, precision, roc_auc, pr_auc = [], [], [], []

            for score in result['result']['scores']:
                accuracy.append(score['accuracy'])
                precision.append(score['precision'])
                roc_auc.append(score['roc_auc'])
                pr_auc.append(score['pr_auc'])

                patient_id = score['patient_id']
                if patient_id not in patients_results:
                    patients_results[patient_id] = {
                        'predictions': [],
                        'accuracy': [],
                        'precision': [],
                        'roc_auc': [],
                        'pr_auc': [],
                    }
                patients_results[patient_id]['predictions'].append(
                    score['predictions'])
                patients_results[patient_id]['accuracy'].append(
                    score['accuracy'])
                patients_results[patient_id]['precision'].append(
                    score['precision'])
                patients_results[patient_id]['roc_auc'].append(
                    score['roc_auc'])
                patients_results[patient_id]['pr_auc'].append(score['pr_auc'])

            segment_results.append({
                'segment': result['result']['segment'],
                'pr_auc': np.mean(pr_auc),
                'classifier': classifier,
                'notes': self.notes,
                'results': pr_auc
            })
            segment_result_report = segment_result_report.append({
                'segment': result['result']['segment'],
                'accuracy': np.mean(accuracy),
                'precision': np.mean(precision),
                'roc_auc': np.mean(roc_auc),
                'pr_auc': np.mean(pr_auc),
                'classifier': classifier,
                'best_estimator': str(result['result']['model']),
                'features': result['result']['final_features'],
                'feature_selection': result['result']['selectors'],
                'timestamp': self.timestamp,
                'dataset': self.dataset.dataset_name,
                'notes': self.notes
            }, ignore_index=True)

        segment_result_report.to_csv(
            f'{self.root_directory}/segment_report.csv')

        return patients_results
