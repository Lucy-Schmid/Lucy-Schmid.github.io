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

1.  Can market indicators help predict the next month's lithium carbonate price?
2.  How do linear and non-linear regression models compare on data not used for training?
3.  How can forecast uncertainty be estimated using Gaussian Process Regression and historical forecast errors?
4.  Can the next month's price be classified into a low, medium, or high price regime?

## Data and Features

The consolidated dataset contains monthly observations. The current modeling workflow uses:

-   Current selected lithium carbonate price
-   Albemarle share price
-   LME copper price
-   LME cobalt cash price (updated Datastream series)
-   CNY/USD exchange rate

Information from month *t* is used to predict the lithium carbonate price in month *t + 1*. Rows with missing values in the selected predictors or next-month target are excluded rather than automatically interpolated. Consequently, the modeling sample is smaller than the original consolidated dataset.

## Methods

### Data preparation and feature engineering

-   Load and sort monthly observations by date.
-   Select the target and market-indicator features.
-   Create a one-month-ahead target.
-   Remove rows that do not contain the required observations.

### Regression models

The notebook compares:

-   Multiple Linear Regression
-   K-Nearest Neighbors Regression
-   Random Forest Regression
-   Gradient Boosting Regression
-   Gaussian Process Regression (GPR)

Selected model hyperparameters are tuned using `GridSearchCV` and chronological `TimeSeriesSplit` cross-validation. A chronological train/test split holds out later observations for final evaluation.

### Forecast uncertainty

Two approaches are explored:

1.  **Gaussian Process Regression:** produces a predictive mean and
    standard deviation, used to construct an approximate 95% predictive
    interval.
2.  **Historical rolling-origin errors:** estimates an empirical
    forecast-error range from sequential predictions made using only
    earlier observations.

These are model-based uncertainty estimates, not guarantees.

### Price-regime classification

A supplementary classification task assigns the next month's price to
one of three categories:

-   **Low**
-   **Medium**
-   **High**

The thresholds are calculated from the training-period target
distribution (terciles), avoiding use of held-out test observations to
define the classes. Logistic Regression and Random Forest classification
are compared.

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