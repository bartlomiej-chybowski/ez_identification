import gc
import os
import pickle
from typing import Optional, List

import numpy as np
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from sklearn.impute import SimpleImputer

from src.suffix_enum import SuffixEnum
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class Dataset:
    """
    A class representing dataset object.

    Attributes
    ----------
    dataset_name : str
        name of the dataset
    engels_outcome : int
        the highest outcome taken into account in Engel's scale.
        Represented as integer. Example: 14 equals 1D, 41 equals 4A
    calculate_features : bool
        flag indicating if variance and average features should be calculated
    dataset: pandas.DataFrame
        DataFrame with data to analyse
    suffix: Optional[List]
        list of suffixes to use. Empty string means features without suffix,
        None uses all features.

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """

    def __init__(self,
                 dataset_name: str,
                 engels_outcome: int = 14,
                 calculate_features: bool = True,
                 suffix: Optional[List] = None):
        """
        Constructs dataset object.

        Parameters
        ----------
        dataset_name : str
            name of the dataset
        engels_outcome : int
            the highest outcome taken into account in Engel's scale.
            Represented as integer. Example: 14 equals 1D, 41 equals 4A
        calculate_features : bool
            flag indicating if variance and average features should be
            calculated
        """
        self.dataset_name = dataset_name
        self.engels_outcome = engels_outcome
        self.calculate_features = calculate_features
        self.suffix = suffix
        self.dataset = self._load_dataset()
        self.segments = self.dataset['segm'].unique()

    def _load_dataset(self) -> pd.DataFrame:
        """
        Loads pickle with dataset.

        Optionally can calculate variance and average features.

        Returns
        -------
        pandas.DataFrame
        """
        df = pickle.load(open(f"input/{self.dataset_name}", "rb"))

        # patient number 756 is a good outcome too, add to analysis
        df = df.loc[(df['outcome'] <= self.engels_outcome) |
                    (df['patient_id'] == 756)]

        if self.calculate_features:
            if 'pathology' not in df.columns.values:
                df.insert(4, "pathology", "normal")
                df.loc[
                    df['onset_channel'] == 'SOZ', 'pathology'] = 'pathologic'
            df = df.loc[:, ~df.columns.isin(['orig_x', 'orig_y', 'orig_z',
                                             'structure', 'sleep_stage',
                                             'night', 'onset_channel'])]

            cols = list(range(4, df.shape[1]))
            numerical_imputer = SimpleImputer(missing_values=np.nan,
                                              strategy='mean')
            numerical_imputer.fit(df.iloc[:, cols])
            df.iloc[:, cols] = numerical_imputer.transform(df.iloc[:, cols])
            df = self._calculate_features(self._generate_segments_labels(df))

        if self.suffix is not None:
            columns = ['patient_id', 'channel_name', 'resected', 'pathology',
                       'outcome', 'segm_type', 'segm']
            if 'segm_number' in df.columns.values:
                columns.append('segm_number')

            if SuffixEnum.RAW not in self.suffix:
                df = df.loc[df['segm'] == '1_0']

            if SuffixEnum.RAW in self.suffix:
                columns.extend([x for x in df.columns
                                if not any(x.endswith(s.value)
                                           for s in
                                           [SuffixEnum.AVG, SuffixEnum.VAR])])
                self.suffix.remove(SuffixEnum.RAW)
            columns.extend([x for x in df.columns if
                            any(x.endswith(s.value) for s in self.suffix)])
            df = df.loc[:, df.columns.isin(list(set(columns)))]

        return df

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates variance and average features.

        Features are calculated for each patient for each channel.

        Example:
        >>> _calculate_features(
                pd.DataFrame(columns=['patient_id', 'channel_name', 'f_1',
                                      'f_2', 'segment']
                             data=[[1, 'a1', 1, 5, '1_0'],
                                   [1, 'a2', 2, 4, '1_0'],
                                   [1, 'a1', 3, 1, '2_0'],
                                   [1, 'a2', 8, 4, '2_0']])
        pd.DataFrame(columns=['patient_id', 'channel_name', 'f_1', 'f_1_var',
                              'f_1_avg', f_2', 'f_2_var', 'f_2_avg', 'segment']
                     data=[[1, 'a1', 1, 1, 2, 5, 4, 3, '1_0'],
                           [1, 'a2', 2, 9, 4, 4, 0, 4, '1_0'],
                           [1, 'a1', 3, 1, 2, 1, 4, 3, '2_0'],
                           [1, 'a2', 8, 9, 4, 4, 0, 4, '2_0']])

        Parameters
        ----------
        df: pandas.DataFrame
            original DataFrame

        Returns
        -------
        pandas.DataFrane
        """
        unique_channels = df.loc[:, 'channel_name'].unique()
        unique_patients = df.loc[:, 'patient_id'].unique()
        df_copy = df.copy()
        for label in df.columns.values[
                     5:(-3 if 'segm_number' in df.columns.values else -2)]:
            df_copy[f"{label}_var"] = 0
            df_copy[f"{label}_avg"] = 0

        inputs = []
        for index, patient in enumerate(df.groupby('patient_id')):
            inputs.append((patient[1], patient[1].copy(), patient[0],
                           unique_channels, index))

        pool = mp.Pool(processes=mp.cpu_count(), initargs=(mp.RLock(),),
                       initializer=tqdm.set_lock)
        jobs = [pool.apply_async(self._calculate_features_for_patient,
                                 args=inputs[i])
                for i in range(len(inputs))]
        pool.close()
        outputs = [job.get() for job in jobs]
        os.system('clear')
        output_df = pd.concat(outputs, axis=0)

        pickle_out = open(f'input/{self.dataset_name[:-4]}_var.pkl', "wb")
        pickle.dump(output_df, pickle_out)
        pickle_out.close()

        return output_df

    @staticmethod
    def _calculate_features_for_patient(df: pd.DataFrame,
                                        df_copy: pd.DataFrame, patient: int,
                                        unique_channels: pd.Series,
                                        pid: int) -> None:
        """
        Calculate average and variance features for each unique channel.

        Parameters
        ----------
        df: pandas.Dataframe
        df_copy: pandas.Dataframe
        patient: int
        unique_channels: pandas.Series
        pid: int

        Returns
        -------
        None
        """
        with tqdm(total=len(unique_channels), desc=f"Patient {patient:4.0f}",
                  position=pid + 1) as pbar:
            for unique_channel in unique_channels:
                last_col = (-3 if 'segm_number' in df.columns.values else -2)
                tmp = df.loc[df['channel_name'] == unique_channel,
                             df.columns[5:last_col]]
                if len(tmp) == 0:
                    pbar.update(1)
                    continue
                for label, content in tmp.items():
                    df_copy.loc[
                        (df['channel_name'] == unique_channel),
                        f"{label}_var"] = content.var()
                    df_copy.loc[
                        (df['channel_name'] == unique_channel),
                        f"{label}_avg"] = content.mean()
                pbar.update(1)
        gc.collect()

        return df_copy

    @staticmethod
    def _generate_segments_labels(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate segments labels.

        Parameters
        ----------
        df: pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        if 'segm_number' in df.columns.values:
            df['segm'] = df.apply(
                lambda x: f"{int(x['segm_type'])}_{int(x['segm_number'])}",
                axis=1)
        else:
            df['segm'] = df.apply(
                lambda x: f"{int(x['segm_type'])}_0", axis=1)

        return df
