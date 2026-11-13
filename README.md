# Lucy-Schmid.github.io
Data Science Project - Lithium Forecasting
# Lucy Schmid — Data Science Projects
Chemical Engineering, University of Florida
---
## Project 1: Predicting Reactor Yield
**The question.** Can we predict yield from temperature
and residence time without running every experiment?
**The data.** 142 pilot-plant runs collected over 2024,
with 3 input variables and 1 measured output.
**What I did.**
1. Cleaned the data and removed 6 sensor faults
2. Fitted a regression model
3. Tested it on data the model had never seen
**What I found.** The model predicts yield to within
3.2% on new data. Temperature mattered far more than
residence time.
### Notebooks
- [1. Cleaning the data](notebooks/01_cleaning.ipynb)
- [2. Building the model](notebooks/02_model.ipynb)
### Results
![Model predictions](figures/predictions.png)
---
## About me
Master's Student in Chemical Engineering interested in
process modelling. Contact: jsmith@ufl.edu