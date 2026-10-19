# Housing Regression Analysis

Two statistical models built on the same housing dataset (7,000 records):

1. **[Multiple Linear Regression](#1-house-price-prediction-multiple-linear-regression):** what predicts a home's sale price?
2. **[Logistic Regression](#2-luxury-home-classification-logistic-regression):** what predicts whether a home is classified as luxury?

*Completed for D600 (Statistical Data Mining), M.S. Data Analytics, Western Governors University.*

---

## Shared Workflow

Both analyses follow the same process:

1. **Exploration:** descriptive statistics, plus univariate and bivariate visualizations
2. **Cleaning:** checked for duplicates, missing values, negative or illogical values, inconsistent categorical labels, data types, and outliers (IQR method)
3. **Encoding:** categorical Yes/No variables converted to binary (0/1)
4. **Train/test split:** 70/30 (4,900 / 2,100 rows), `random_state=100`
5. **Optimization:** backward stepwise elimination, removing the predictor with the highest p-value one at a time until every remaining predictor had p < 0.05
6. **Evaluation:** performance measured on the held-out test set

---

## 1. House Price Prediction (Multiple Linear Regression)

**Question:** Which characteristics are the strongest predictors of a home's sale price?

**Candidate predictors (10):** `SquareFootage`, `NumBedrooms`, `NumBathrooms`, `Garage`, `AgeOfHome`, `Floors`, `SchoolRating`, `CrimeRate`, `DistanceToCityCenter`, `LocalAmenities`

**Cleaning notes:** Outliers in `SquareFootage`, `AgeOfHome`, `CrimeRate`, and `DistanceToCityCenter` were capped at the IQR bounds. `Floors` and `SchoolRating` were flagged by IQR but kept, as the values were within logical ranges.

| Step | Removed | p-value |
|---|---|---|
| 1 | CrimeRate | 0.997 |
| 2 | Garage | 0.886 |
| 3 | Floors | 0.833 |
| 4 | SchoolRating | 0.089 |

**Final model (6 predictors, all p < 0.05):**

```
Price = -133,651.45
        + 159.76    × SquareFootage
        + 50,477.30 × NumBathrooms
        + 56,303.30 × NumBedrooms
        - 161.91    × AgeOfHome
        - 561.77    × DistanceToCityCenter
        + 2,536.31  × LocalAmenities
```

| Metric | Initial model (10 predictors) | Optimized model (6 predictors) |
|---|---|---|
| Adjusted R² | 0.592 | 0.592 |
| Test RMSE | — | ~$97,802 |

**Key findings:**
- The model explains about 59% of the variance in price. Dropping four predictors didn't reduce explanatory power.
- On the test set, predictions are typically off by roughly $98K, about 32% of the mean price ($307,282).
- Square footage, bedrooms, bathrooms, and local amenities are associated with **higher** prices. Age and distance from the city center are associated with **lower** prices.
- Coefficients are per one-unit change, so they aren't directly comparable in magnitude. For example, a bedroom adds about $56K, and 500 sq ft adds about $80K.

**Assumptions & limitations:** Linearity and absence of multicollinearity were checked. The residuals show heteroscedasticity: they spread out as predicted price increases, so the model is less reliable for higher-priced homes.

---

## 2. Luxury Home Classification (Logistic Regression)

**Question:** Which variables are the strongest predictors of a home being classified as luxury (`IsLuxury`)?

**Candidate predictors (10):** `SquareFootage`, `NumBathrooms`, `NumBedrooms`, `RenovationQualityRate`, `LocalAmenities`, `TransportAccess`, `SchoolRating`, `DistanceToCityCenter`, `Fireplace`, `Garage`

**Cleaning notes:** `RenovationQualityRate`, `TransportAccess`, and `SchoolRating` were flagged by IQR but kept, as the values were within logical ranges.

| Step | Removed | p-value |
|---|---|---|
| 1 | Garage | 0.711 |
| 2 | Fireplace | 0.280 |
| 3 | SchoolRating | 0.232 |
| 4 | DistanceToCityCenter | 0.174 |
| 5 | TransportAccess | 0.077 |

**Final model (5 predictors, all p < 0.05):**

```
logit(p) = -5.8159
           + 0.0019 × SquareFootage
           + 0.5535 × NumBathrooms
           + 0.6705 × NumBedrooms
           + 0.1004 × RenovationQualityRate
           + 0.0445 × LocalAmenities
```

| Predictor | Odds ratio | Effect on odds of luxury |
|---|---|---|
| SquareFootage | 1.002 | +0.2% per sq ft |
| NumBathrooms | 1.74 | +74% per bathroom |
| NumBedrooms | 1.96 | +96% per bedroom |
| RenovationQualityRate | 1.11 | +11% per point |
| LocalAmenities | 1.05 | +5% per point |

| Metric | Initial model (10 predictors) | Optimized model (5 predictors) |
|---|---|---|
| Pseudo R² | 0.2272 | 0.2261 |
| Test accuracy | — | 74.76% |

**Confusion matrix (test set, n = 2,100):**

| | Predicted: Not luxury | Predicted: Luxury |
|---|---|---|
| **Actual: Not luxury** | 778 | 281 |
| **Actual: Luxury** | 249 | 792 |

**Key findings:**
- The model correctly classifies about 3 out of 4 homes, and errors are fairly balanced in both directions.
- The test set is close to evenly split (1,059 non-luxury / 1,041 luxury), so 74.76% accuracy is a meaningful improvement over guessing.
- Removing five predictors had almost no effect on pseudo R².

**Assumptions & limitations:** Multicollinearity was checked with VIF (all values below 1.4). Linearity of the log-odds was checked with the Box-Tidwell test: `RenovationQualityRate` failed it (p = 0.014), so its coefficient should be interpreted with caution, because its effect may not be constant across the full rating range.

---

> **Disclaimer:** These findings are specific to this dataset. The predictors identified as most influential, and their effect sizes, may not hold for other datasets, markets, or time periods. Both models work better as guides to **which features matter** than as precise pricing or classification tools.

## Tools

Python · pandas · NumPy · matplotlib · statsmodels · scikit-learn


