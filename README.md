# Tindihan ni BERB: E-Commerce Analytics (CSE315 Final Project)

CRISP-DM analysis of one year of online-store orders (UCI Online Retail, Dec 2010 – Dec 2011) for **Tindihan ni BERB**, a fictional gift and party-supply e-commerce store in Quezon City.

**Team (BSCS 3A, Gordon College):** John Melchor Brillante · Rylan Bernardino · Adrian Raymundo · Alan Jr. P. Ecalnir

## Research questions and headline results

| RQ | Question | Method | Result |
|---|---|---|---|
| RQ1 | Which products are bought together? | Apriori / FP-Growth, network graph | 54 rules, mean lift 12.3; 8 product families; hubs: Lunch Bag Pink Polkadot, Lunch Bag Red Retrospot |
| RQ2 | What types of customers are there? | RFM + k-Means (k = 4) | Loyal / Big Spenders = 15.8% of customers, 64.0% of revenue; segments stable (ARI 0.974) |
| RQ3 | When do sales happen? | Monthly trend, linear regression | Peak Sep–Nov 2011 (≈1.14M in Nov); spend driven by product variety (test R² 0.877) |
| RQ4 | Who is likely to cancel? | LogReg, LDA, Decision Tree, MLP | Balanced Logistic Regression: test ROC-AUC 0.771, recall 0.704, F1 0.637 |

## Repository layout

```
notebooks/
  02_model_pipeline.ipynb        # Phase 2: all models, experiment log, model file
scripts/
  make_portfolio_charts.py       # Phase 3: rebuilds the visualization portfolio
  style.py                       # shared chart style and colour palette
outputs/
  Model_Experiment_Log.xlsx/.csv # 60 logged runs (ID, author, date, settings, metrics)
  model_artifacts.pkl            # scaler, k-Means, rules, classifier, regression
  top_pairs.csv, network_clusters.csv, classifier_runs.csv
figures/                         # visualization portfolio (PNG, 200 dpi)
  phase1_2_originals/            # charts exported from the Phase 1 report and Phase 2 notebook
reports/
  Tindihan_ni_BERB_Final_Report.pdf
  Phase1_Business_Data_Preparation.pdf
docs/
  pitch_script.md                # 5–7 minute boardroom briefing script
data/                            # raw data goes here (not committed)
```

## How to reproduce

```bash
pip install -r requirements.txt
# 1. Phase 1 (cleaning, see reports/Phase1_Business_Data_Preparation.pdf) produces
#    basket_matrix.csv, customer_rfm.csv, customer_rfm_scaled.csv, cancellations.csv in data/
# 2. Run notebooks/02_model_pipeline.ipynb (writes outputs/)
# 3. Rebuild the portfolio charts:
python scripts/make_portfolio_charts.py
```

All random steps use `random_state = 42`. The Phase 2 notebook ends with 12 PASS/FAIL self-checks.

### Using the saved models

```python
import pickle, numpy as np, pandas as pd
A = pickle.load(open("outputs/model_artifacts.pkl", "rb"))
c = pd.DataFrame({"Recency": [10], "Frequency": [8], "Monetary": [5000.0]})
seg = A["kmeans"].predict(A["scaler_rfm"].transform(np.log1p(c[A["rfm_features"]])))[0]
print(A["cluster_names"][seg])   # Loyal / Big Spenders
```

Load with scikit-learn 1.6.1 and mlxtend 0.23.4 (the versions used to save it), or retrain from the notebook.

## Data and privacy

Data: Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33

The data holds only numeric customer IDs (no names or contact details). Customer-level tables are not committed. Model scores are meant for helpful actions such as order confirmations, never for refusing service. See Section 7 of the final report.
