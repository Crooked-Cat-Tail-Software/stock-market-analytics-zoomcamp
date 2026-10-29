"""Homework 4 runner: executes the official Module 04 notebook headless, with per-question edits.

Usage: python hw4.py <baseline|q1|q2|q3> <path-to-official-notebook.ipynb>

Each variant starts from a fresh copy of the official notebook (as the homework requires),
applies only that question's edits, and writes results to hw4_results/<variant>_*.csv.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'content', 'stocks_df_combined_2026_09_18.parquet.brotli')
OUT = os.path.join(HERE, 'hw4_results')

# Q1: drop Ticker and month_wom from CATEGORICAL just before pd.get_dummies()
Q1_PATCH = ("CATEGORICAL.append('month_wom')\nCATEGORICAL",
            "CATEGORICAL.append('month_wom')\n"
            "CATEGORICAL = [c for c in CATEGORICAL if c not in ('Ticker', 'month_wom')]\nCATEGORICAL")

# Q2: tpr_fpr_dataframe on Random Forest instead of the Decision Tree
Q2_PROBA_PATCH = ("y_pred_test = clf_best.predict_proba(X_test)\n\n# y_pred_test = rf_best.predict_proba(X_test)",
                  "# y_pred_test = clf_best.predict_proba(X_test)\n\ny_pred_test = rf_best.predict_proba(X_test)")

# Q2: extra RF decision rules for thresholds 0.56..0.64
Q2_RULES_ANCHOR = "new_df['pred11_rf_best_rule_65'] = (y_pred_all_class1_array >= 0.65).astype(int)"
Q2_RULES = Q2_RULES_ANCHOR + "\n" + "\n".join(
    f"new_df['pred{12 + i}_rf_best_rule_{t}'] = (y_pred_all_class1_array >= 0.{t}).astype(int)"
    for i, t in enumerate(range(56, 65)))

# Q3: redefine the growth target (right after loading the data, so all later steps use it)
Q3_ANCHOR = 'df_full = pd.read_parquet("/content/stocks_df_combined_2026_09_18.parquet.brotli", )'
Q3_PATCH = Q3_ANCHOR + "\ndf_full['is_positive_growth_30d_future'] = (df_full['growth_future_30d'] > 1.022).astype(int)"
# Q3: rule_65 has no positive test predictions with the new target, so this display-only cell can't sample 10 rows
Q3_SAMPLE_ANCHOR = "new_df[(new_df.split=='test')&(new_df.pred11_rf_best_rule_65==1)].sort_values(by='Date').sample(10)"
Q3_SAMPLE_PATCH = "new_df[(new_df.split=='test')&(new_df.pred11_rf_best_rule_65==1)].sort_values(by='Date').head(10)"


def replace_once(src, old, new):
    if src.count(old) != 1:
        raise ValueError(f'patch anchor found {src.count(old)} times: {old[:60]!r}')
    return src.replace(old, new)


def build_script(nb_path, variant):
    nb = json.load(open(nb_path))
    cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']

    # Neural-network section only uses pre-computed history and feeds no predictor: skip it (no TensorFlow needed)
    start = next(i for i, s in enumerate(cells) if "p = {'lr'" in s)
    end = next(i for i, s in enumerate(cells) if 'Training vs. Test Precision' in s)
    cells = cells[:start] + cells[end + 1:]

    src = '\n#----CELL----\n'.join(cells)
    if variant == 'q3':
        src = replace_once(src, Q3_ANCHOR, Q3_PATCH)
        src = replace_once(src, Q3_SAMPLE_ANCHOR, Q3_SAMPLE_PATCH)
    if variant in ('q1', 'q2'):
        src = replace_once(src, *Q1_PATCH)
    if variant == 'q2':
        src = replace_once(src, *Q2_PROBA_PATCH)
        src = replace_once(src, Q2_RULES_ANCHOR, Q2_RULES)
    src = src.replace('"/content/stocks_df_combined_2026_09_18.parquet.brotli"', repr(DATA))

    lines = []
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith(('!', '%')):  # shell / magics
            continue
        if stripped in ('fig.show()', 'plt.show()'):
            line = line.replace(stripped, "plt.close('all')")
        lines.append(line)
    return '\n'.join(lines)


def main():
    variant, nb_path = sys.argv[1], sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    script = build_script(nb_path, variant)
    ns = {}
    exec(compile(script, f'module04_{variant}', 'exec'), ns)

    n_features = len(ns['NUMERICAL'] + ns['DUMMIES'])
    pd = ns['pd']
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 20)
    res = ns['df_sim1_results']
    res.to_csv(os.path.join(OUT, f'{variant}_sim1_results.csv'), index=False)
    ns['df_scores'].to_csv(os.path.join(OUT, f'{variant}_df_scores.csv'), index=False)

    print(f'\n===== {variant}: features (NUMERICAL+DUMMIES) = {n_features}, '
          f'class-1 share = {ns["new_df"].is_positive_growth_30d_future.mean():.3f}')
    print(res[['prediction', 'sim1_count_investments', 'sim1_net_rev', 'sim1_avg_investments_per_day',
               'sim1_capital', 'sim1_CAGR']].to_string())
    if variant == 'q2':
        print(ns['df_scores'][(ns['df_scores'].threshold >= 0.5) & (ns['df_scores'].threshold <= 0.7)].to_string())


if __name__ == '__main__':
    main()
