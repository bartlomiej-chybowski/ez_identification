# Epileptogenic zone identification

[![DOI](https://zenodo.org/badge/DOI/DOI.NUMBER/zenodo.NUMBER.svg)](https://doi.org/DOI.NUMBER/zenodo.NUMBER)
[![Project Status: Concept – Minimal or no implementation has been done yet, or the repository is 
only intended to be a limited example, demo, or proof-of-concept.](https://www.repostatus.org/badges/latest/concept.svg)](https://www.repostatus.org/#concept)

Analyse EEG data.

## Abstract

**Objective:** Interictal biomarkers of the epileptogenic zone (EZ) and their use in machine learning models open promising avenues for improvement of epilepsy surgery evaluation. Currently, most studies restrict their analysis to short segments of intracranial EEG (iEEG). Therefore, the effect of sleep and circadian rhythm on these models are not well understood despite increasing evidence that these markers undergo significant spatio-temporal fluctuations throughout the day. \
**Methods:** We used 2381 hours of iEEG data from 25 patients to systematically select 5-minute segments across various sleep stages and interictal conditions. Then, we tested machine learning models for EZ localization using iEEG features calculated within these individual segments or across them and evaluated the performance by the area under the precision-recall curve (PRAUC). \
**Results:** On average, models achieved a score of 0.421 (the result of the chance classifier was 0.062). However, the PRAUC varied significantly across the segments (0.323-0.493). Overall, NREM sleep achieved the highest scores, with the best results of 0.493 in N2. When using data from all segments, the model performed significantly better than single segments, except NREM sleep segments. \
**Interpretation:** The model based on a short segment of iEEG recording can achieve similar results as a model based on prolonged recordings. The analyzed segment should, however, be carefully and systematically selected, preferably from NREM sleep. Random selection of short iEEG segments may give rise to inaccurate localization of the EZ.

## How to install

In order to run the pipeline, please run the following commands:
```bash
# install project dependencies
poetry install
# run project
python main.py
```
