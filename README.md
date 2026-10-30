# stock-market-analytics-zoomcamp

## Homework 4: Trading System

Assignment: [homework4.md](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/cohorts/2026/homework4.md)

Each question starts from a fresh copy of the official Module 04 notebook, as the assignment requires.

| Question | Notebook | Answer |
|---|---|---|
| Q1: CAGR for `pred10_rf_best_rule_55` with fewer dummies | [q1 notebook](%5B2026%5D_Module_04_Colab_Trading_Simulations_q1.ipynb) | 1.126 |
| Q2: second-best RF threshold with < 5 trades/day | [q2 notebook](%5B2026%5D_Module_04_Colab_Trading_Simulations_q2.ipynb) | 0.57 |
| Q3: avg. trades per day for `pred5_clf_10` (target > 1.022) | [q3 notebook](%5B2026%5D_Module_04_Colab_Trading_Simulations_q3.ipynb) | 8 |
| Q4: ideal trading strategy | — | TODO: write in your own words |

### Edits per question
- **Q1:** removed `Ticker` and `month_wom` from `CATEGORICAL` just before `pd.get_dummies()` (301 → 208 features).
- **Q2:** built on Q1; `tpr_fpr_dataframe` uses `rf_best.predict_proba`, and added RF rules for thresholds 0.56–0.64 (`pred12`–`pred20`).
- **Q3:** redefined `is_positive_growth_30d_future = growth_future_30d > 1.022` (~49% class 1, down from ~59%).

### Note on environments
The local notebooks ran with newer pandas/scikit-learn, which shifts the Random Forest slightly (local Q1 CAGR 1.1189, Q2 threshold 0.59). [hw4.py](hw4.py) reruns each question headless in a Colab-matched environment (Python 3.12, pandas 2.2.2, scikit-learn 1.6.1), which gives Q1 1.1291 and Q2 0.57, matching the form's options. Q3 (a single Decision Tree) is identical in both: 7201 trades, 7.85 per day. Results are in [hw4_results/](hw4_results/).
