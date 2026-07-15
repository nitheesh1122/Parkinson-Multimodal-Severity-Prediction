# Parkinson Speech Dataset Analysis

## Dataset Size

- Total instances: 195 voice recordings
- Subjects: 31 people
- Parkinson's cases: 23 subjects
- Healthy controls: 8 subjects

## Features

The dataset contains biomedical voice measurements, including:

- Fundamental frequency measures: `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`
- Jitter measures: `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `Jitter:DDP`
- Shimmer measures: `MDVP:Shimmer`, `Shimmer:DDA`, `APQ3`, `APQ5`
- Noise and harmonic measures: `NHR`, `HNR`
- Nonlinear measures: `RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE`
- Label column: `status` (1 = PD, 0 = Healthy)

## Missing Values

- Missing values are generally reported as none in the cleaned UCI version.
- Validation step recommended in preprocessing:
  - null check per feature,
  - duplicate sample check,
  - value range screening for biomedical plausibility.

## UPDRS Context

- The core UCI Parkinson Speech dataset used for classification does not directly include total UPDRS scores.
- For severity-oriented extensions, UPDRS-linked datasets (for example telemonitoring variants) are required.
- UPDRS can be integrated as:
  - target for regression or ordinal severity grouping,
  - reference score in clinician-facing reports.

## Sample Distribution

- Class distribution is imbalanced toward PD samples.
- Subject-level distribution should be preserved during split to avoid data leakage.
- Recommended split strategy:
  - grouped train/validation/test split by subject,
  - imbalance handling using class weights or SMOTE variants.
