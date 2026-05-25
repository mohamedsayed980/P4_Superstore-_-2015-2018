cat > /mnt/user-data/outputs/README.md << 'EOF'
# 🛍️ Superstore Sales — End-to-End Data Analysis & ML
> **Author:** Mohamed · M3 · Data Analysis Portfolio  
> **Dataset:** Tableau Superstore Sales (Kaggle)  
> **Stack:** Python · Streamlit · Scikit-learn · Pandas · Matplotlib · Seaborn

---

## 📌 Project Overview

A full end-to-end Business Intelligence and Machine Learning project on the **Tableau Superstore dataset** — 9,994 real US retail orders across 4 regions, 3 categories, and 17 sub-categories from 2015 to 2018.

> **Key finding:** 19.4% of all orders are unprofitable — losing up to $6,600 per order. Discounts above 20% consistently destroy profit margins.

---

## 🗂️ Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle / Tableau Public |
| File | `Sample - Superstore.csv` |
| Shape | 9,994 rows × 20 columns |
| Missing Values | None |
| Period | 2015 – 2018 |
| Regions | 4 US regions |

### Dataset Prep (Jupyter)
```python
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")
df.columns = df.columns.str.strip()  # removes hidden spaces
df["delivery_days"] = (df["Ship Date"] - df["Order Date"]).dt.days
df["profit_margin"] = df["Profit"] / (df["Sales"] + 0.01)
df["is_profitable"] = (df["Profit"] > 0).astype(int)
df["is_discounted"] = (df["Discount"] > 0).astype(int)
df.to_csv("superstore_clean.csv", index=False)
```

---

## 🏗️ Architecture

```
📁 Repo_4_Superstore/
├── Home.py                    ← Multipage launcher
├── M3_logo.png
├── requirements.txt
├── README.md
├── pages/
│   ├── EDA_dashboard.py       ← Stage 1 (Tabs 1–13)
│   └── ML_Models.py           ← Stage 2 (Tabs 9–13)
└── data/
    └── superstore_clean.csv
```

**Run:** `streamlit run Home.py`

---

## 📊 Stage 1 — EDA Dashboard (13 Tabs)

| Tab | Name | Description |
|-----|------|-------------|
| 1 | Data Overview & Correlation | Shape, stats, top correlations |
| 2 | Variables Analysis | Distributions, histograms |
| 3 | IQR Cleaning | Outlier detection |
| 4 | Outliers Lab | Z-score, box plots |
| 5 | Dashboard Summary | Visual summary |
| 6 | Missing Values | Heatmap, imputation |
| 7 | Multicollinearity | VIF analysis |
| 8 | Insights | Auto-generated insights |
| 9 | **Business KPI Dashboard** ⭐ | Sales, profit, margin, segment |
| 10 | **Profit & Loss Analysis** ⭐ | Discount impact, sub-category P&L |
| 11 | **Regional Performance** ⭐ | Interactive metric by region/state/city |
| 12 | **Time Series Trends** ⭐ | Monthly/annual trends, YoY growth |
| 13 | **Statistical Tests** ⭐ | ANOVA, T-Test, Chi-Square |

---

## 🤖 Stage 2 — ML Models (5 Tabs)

| Tab | Name | Description |
|-----|------|-------------|
| 9 | Regression Models (6) | Predict **Profit** |
| 10 | Classification Models (6) | Predict **is_profitable** (0/1) |
| 11 | Comparison & Report | Model comparison |
| 12 | Predict New Order | Input features → prediction |
| 13 | Final Insights & Report | PDF + Word export |

### Models
**Regression (6):** Linear · Ridge · Lasso · Decision Tree · Random Forest · Gradient Boosting

**Classification (6):** Logistic Regression · KNN · Decision Tree · Random Forest · Gradient Boosting · SVM

---

## 🔑 Key Business Insights

- **19.4% of orders unprofitable** — losing up to $6,600 per order
- **Discounts > 20% destroy margins** — avg profit turns negative above 20% discount
- **Technology = highest revenue** but Furniture has worst profit margin
- **Tables sub-category** = biggest loss maker across all regions
- **West region** = highest sales · **South** = lowest profit rate
- **YoY growth:** consistent upward trend 2015→2018
- **Statistical confirmation:** Discounts significantly reduce profit (T-Test p < 0.05)

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/Repo_4_Superstore.git
cd Repo_4_Superstore
pip install -r requirements.txt
streamlit run Home.py
```

## 📥 Dataset Setup
1. Download from [Kaggle — Superstore](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
2. Run Jupyter prep → produces `superstore_clean.csv`
3. Upload via app file uploader

---

## 🗺️ Portfolio Roadmap

| # | Project | Domain | Status |
|---|---------|--------|--------|
| P1 | House Prices Analysis | Real Estate | ✅ Complete |
| P2 | Olist E-Commerce | Retail / Business | ✅ Complete |
| P3 | HR Attrition | Human Resources | ✅ Complete |
| P4 | Superstore Sales | Business Intelligence | ✅ Complete |
| P5–P7 | Coming Soon... | Finance · Environment · Fraud | 🔜 |

---

## 👤 Author
**Mohamed · M3** — Mechanical Engineer → Data Analyst  
📊 Building a professional DA/DS portfolio — one dataset at a time.

*Built with ❤️ using Python & Streamlit*
EOF
echo "✅ README.md done"
Output

✅ README.md done
