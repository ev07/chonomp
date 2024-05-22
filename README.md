# ChronoEpilogi: Scalable time-series variables Selection with Multiple Solutions

Codebase for the under-review paper of the same name.

## Abstract

We consider the problem of selecting all the minimal-size subsets of multivariate
time-series (TS) variables whose past leads to an optimal predictive model for the
future (forecasting) of a given target variable (multiple feature selection problem for
times-series). Identifying these subsets leads to gaining insights, domain intuition,
and a better understanding of the data-generating mechanism; it is often the first
step in causal modeling. While identifying a single solution to the feature selection
problem suffices for forecasting purposes, identifying all such minimal-size, op-
timally predictive subsets is necessary for knowledge discovery and important to
avoid misleading a practitioner.

We develop the theory of multiple feature selection for time-series data, propose
the ChronoEpilogi algorithm, and prove its soundness and completeness under two
mild, broad, non-parametric distributional assumptions, namely Compositionality
of the distribution and Interchangeability of time-series variable in solutions. Exper-
iments on synthetic and real datasets demonstrate the scalability of ChronoEpilogi
to thousands of TS variables and its efficacy in identifying multiple solutions. In the
real datasets, ChronoEpilogi is shown to reduce the number of TS variables by 96%
(on average) without a significant drop in forecasting performance. Furthermore,
ChronoEpilogi is on par with group Lasso performance, with the added benefit of
providing multiple solutions.

## Exemple Usage



## Code structure

 - Implementation of the proposed algorithm is in the main directory, under `ChronoEpilogi.py`. Subroutines are implemented in `associations.py` and `models.py`.
 - Baselines and Forecasters are wrapped respectively in `\baselines\feature_selection.py` and `\baselines\estimators.py`. Wrappers allows for standard call structure and easier pipeline declaration across libraries.
 - Datasets generation assets are provided in the `\data\` folder. Dataset opening routine is implemented in `data_opener.py`
 - Tuning of the algorithms is implemented in folder `\tuning\`.
   - `first_wave_main.py` implements paired tuning of feature selection algorithm and forecaster.
   - `second_wave_main.py` implements tuning of a forecaster for an already tuned feature selection algorithm.
   - `routines.py` implement shared elements between tuning routines.
 - Final evaluation on a test set is implemented in folder `\testing\`:
   - `results.py` and `results_other.py` implement respectively ChronoEpilogi evaluation and other fs algorithms evaluation.
   - `final_statistics.py` implement tuning record opening routines


