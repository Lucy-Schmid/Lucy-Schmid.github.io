# Lucy Schmid — Data Science Project
Chemical Engineering, University of Florida
---
## Term Project: Predicting Lithium Carbonate Prices Using Machine Learning: A Comparison of Regression Algorithms and Predictive Uncertainty
**The question.** How accurately can machine learning algorithms forecast lithium carbonate prices using historical commodity prices and macroeconomic indicators, and how does predictive uncertainty vary across market conditions?

## Motivation

Lithium prices historically have gone through periods of ups and downs. The main drivers behind this volatility include global economic conditions, commoditiy prices, GDP, currency exchange rates, EV demand, mine supply, and unpredictable phenomena such as pandemics and wars. This market dynamic poses serious challenges for battery manufacturers and smaller miners as lower prices make it harder to survive due to lower profit margins. Accurate price forecasting can aid in hedging, pricing decisions, and production planning.

This project compares various machine learning methods to determine the optimal approach for lithium carbonate price prediction.

## Project Overview

This project investigates whether monthly market indicators can be used to forecast the next month's lithium carbonate price in China. The target variable is **Selected lithium carbonate price (CNY/metric tonne)**.

The analysis combines historical lithium carbonate prices with selectedmarket and macroeconomic indicators. The primary objective is to compare regression models for one-month-ahead price forecasting and evaluate the uncertainty associated with those forecasts.

A supplementary classification task groups forecast prices into **low**, **medium**, and **high** regimes based on the price distribution in the training data.

## Research Questions

1. Can historical lithium carbonate prices and broader market indicators help predict future lithium carbonate prices at 1-, 3-, 6-, and 12-month forecasting horizons?
2. How do univariate and multivariate forecasting models compare in their ability to predict lithium carbonate prices on unseen data?
3. How does reducing predictor complexity and tuning model hyperparameters using time-series cross-validation affect forecasting performance and model generalization?
4. How can forecast uncertainty be estimated using historical prediction errors and approximate prediction intervals?
5. Can the next month's lithium carbonate price be classified into a low, medium, or high price regime using historical price information?

## Data and Features

The consolidated dataset contains monthly observations of lithium carbonate prices and related commodity, financial, battery-market, and macroeconomic indicators. The forecasting workflow uses the selected Chinese lithium carbonate spot price as the target variable:

Target: Selected lithium carbonate price (CNY/t).

The univariate models use historical lithium carbonate prices as their only predictors, incorporating the previous 12 months of price observations.
The consolidated dataset contains monthly observations. The current modeling workflow uses:

-   Current selected lithium carbonate price
-   Lithium hydroxide prices
-   LME copper prices
-   Updated LME cobalt cash prices
-   LME nickel prices
-   Tesla, Albemarle and SQM share prices
-   CNY/USD exchange rate
-   China and U.S. macroeconomic indicators
-   Electric vehicle and battery-market indicators
-   Japanese battery production, shipments, and inventory indicators
-   Other available commodity and lithium-market indicators

The multivariate predictor selection requires at least 60 non-missing monthly observations for each base predictor. The three shorter-history Japanese battery series for inventory, shipments, and production are excluded because they duplicate their corresponding longer-history series over their overlapping observations. The longer-history versions are retained.

The Japanese lithium carbonate import-price index is evaluated separately as an optional additional predictor. This series is an import-price index (2020 = 100), rather than an absolute spot price in CNY/t.

To reduce model complexity and potential overfitting, the multivariate models use three months of historical lagged values for the target and explanatory variables. The univariate models retain 12 months of historical target-price lags.

Information available in previous months is used to predict future lithium carbonate prices. Missing predictor values are handled through median imputation within the model pipelines, with imputation fitted using training data only. Missing target observations are not imputed.

The dataset is divided chronologically into training and testing periods, preserving the time order of observations. The final test period is reserved for evaluating model performance on unseen data.


Information from month *t* is used to predict the lithium carbonate price in month *t + 1*. Rows with missing values in the selected predictors or next-month target are excluded rather than automatically interpolated. Consequently, the modeling sample is smaller than the original consolidated dataset.

## Methods

## Methods

### Data preparation and feature engineering

The monthly dataset is cleaned and prepared by identifying the selected lithium carbonate price as the forecasting target, removing duplicate target representations and faulty cobalt data, and selecting numeric explanatory variables.

Predictors are screened based on historical data availability, requiring at least 60 non-missing monthly observations. Duplicate Japanese battery indicators are consolidated by retaining the longer-history series.

Lagged features are generated to capture historical price and market relationships. The univariate models use 12 months of historical target-price observations, while the multivariate models use three months of historical target-price and explanatory-variable observations.

### Model development and comparison

Several algorithms are evaluated:

- Linear Regression
- K-Nearest Neighbors (KNN)
- Support Vector Regression (SVR)
- Random Forest Regression
- Gradient Boosting Regression
- Gaussian Process Regression (GPR)

To reduce overfitting, model complexity is constrained through hyperparameter settings, including tree depth, minimum leaf size, number of neighbors, and learning rate. Hyperparameters are tuned using chronological time-series cross-validation within the training data.

Univariate and multivariate models are evaluated at 1-, 3-, 6-, and 12-month forecasting horizons.

### Model evaluation

Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

The models are evaluated on a final chronological test set that is not used during hyperparameter selection.

### Forecast uncertainty

Forecast uncertainty is estimated using historical model residuals to construct approximate prediction intervals. Empirical interval coverage is compared with the nominal 95% target to assess how well the intervals capture actual prices.

### Price regime classification

The next-month lithium carbonate price is also classified into low, medium, and high price regimes. The categories are defined using the 33rd and 67th percentiles of the training-period price distribution to avoid using future test information when establishing the thresholds.

Logistic Regression and Random Forest classifiers are evaluated using accuracy, macro F1-score, classification reports, and confusion matrices.

## Repository Contents

Repository structure:

Lucy-Schmid.github.io/
│
├── Data/
│   └── lithium_consolidated_monthly.csv
│
├── Results/
│   ├── figures/
│   └── tables/
│
├── notebooks/
│   ├── Cleaning and Prep/
│   │   ├── 01_Data_Cleaning_and_Exploration.ipynb
│   │   └── 02_Correlation_and_Feature_Selection.ipynb
│   │
│   └── Modelling and Comparison/
│       ├── 03_Univariate_Forecasting.ipynb
│       ├── 04_Multivariate_Forecasting.ipynb
│       └── 05_Model_Comparison_and_Uncertainty.ipynb
│
├── src/
│   ├── features.py
│   └── utils/
│       └── logger.py
│
├── README.md
└── requirements.txt


## How to Run

1.  Clone or download this repository.

2.  Install Python and Jupyter Notebook or JupyterLab.

3.  Install the required packages:

    ``` bash
    pip install numpy pandas matplotlib scikit-learn jupyter
    ```

4.  Place the CSV dataset in the location expected by the notebook.

5.  Open
    `Lithium_Carbonate_Forecasting.ipynb`.

6.  Run the notebook from top to bottom.

## Evaluation

Regression models are evaluated on the held-out test period using:

-   Root Mean Squared Error (RMSE)
-   Mean Absolute Error (MAE)
-   Coefficient of determination (R²)

Classification is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix. Model performance should be compared using held-out chronological observations rather than training performance alone.

## Limitations

-   The complete-case modeling dataset is relatively small, which can make model rankings and uncertainty estimates unstable.
-   The lithium carbonate price series combines source segments; changes in data source may introduce differences in price levels.
-   Some market indicators may not have been publicly available at the exact forecast date because of reporting or publication delays.
-   Monthly market observations can be serially dependent, so random train/test splitting would risk information leakage.
-   Low, medium, and high categories are relative to the training-period price distribution, not fixed industry price thresholds.
-   Gaussian Process intervals depend on the selected kernel and modeling assumptions. Historical error bands assume past forecast errors are informative about future errors.

Results should be interpreted as model-based estimates rather than definitive predictions of lithium market prices.

## Acknowledgments

This project was completed as part of **ECH 6845: Chemical Process Data
Science** at the University of Florida. Special thank you to Dr. Sumant Patankar!

------------------------------------------------------------------------

*Project status: analysis and model comparison in progress. Numerical results will be added after the notebook is run and final model

---
## About me
Master's Student in Chemical Engineering interested in metals & mining. Contact: schmidlucy@ufl.edu