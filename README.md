# House Price Prediction with Multiple Linear Regression

**Question:** Which characteristics are the strongest predictors of a home's sale price?

**Goal:** Identify the features with the most influence on sale price to help realtors and sellers set listing prices.

*Completed for D600 (Statistical Data Mining), M.S. Data Analytics, Western Governors University.*

---

## Data

- 7,000 home sale records, with `Price` as the target
- 10 candidate predictors: `SquareFootage`, `NumBedrooms`, `NumBathrooms`, `Garage`, `AgeOfHome`, `Floors`, `SchoolRating`, `CrimeRate`, `DistanceToCityCenter`, `LocalAmenities`

## Approach

1. **Exploration:** descriptive statistics, plus univariate and bivariate visualizations
2. **Cleaning:** checked for duplicates, missing values, negative or illogical values, inconsistent categorical labels, data types, and outliers (IQR method)
   - Outliers in `SquareFootage`, `AgeOfHome`, `CrimeRate`, and `DistanceToCityCenter` were capped at the IQR bounds
   - `Floors` and `SchoolRating` were flagged by IQR but kept, as the values were within logical ranges.
3. **Encoding:** `Garage` was converted to binary (0/1)
4. **Train/test split:** 70/30 (4,900 / 2,100 rows), `random_state=100`
5. **Modeling:** OLS multiple linear regression with `statsmodels`
6. **Optimization:** backward stepwise elimination, removing the predictor with the highest p-value one at a time until every remaining predictor had p < 0.05

| Step | Removed | p-value |
|---|---|---|
| 1 | CrimeRate | 0.997 |
| 2 | Garage | 0.886 |
| 3 | Floors | 0.833 |
| 4 | SchoolRating | 0.089 |

## Results

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

- The model explains about 59% of the variance in price. Dropping four predictors didn't reduce explanatory power, so the simpler model is preferred.
- On the test set, predictions are typically off by roughly $98K, about 32% of the mean price ($307,282).

## Interpretation

- Square footage, bedrooms, bathrooms, and local amenities are associated with **higher** prices. Age and distance from the city center are associated with **lower** prices.
- Coefficients are per one-unit change, so they aren't directly comparable in magnitude. For example, a bedroom adds about $56K, and 500 sq ft adds about
