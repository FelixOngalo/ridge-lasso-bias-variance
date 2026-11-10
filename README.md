# Ridge, Lasso and the Bias-Variance Trade-off

Ridge and lasso regression implemented from scratch in R, with the underlying
mathematics proved and then verified numerically on simulated data where the
true coefficients are known.

## Question

When there are many correlated predictors and few observations, how much does
shrinkage help, why does it help, and how well can the lasso recover which
variables really matter?

## Setup

- n = 100 observations, p = 50 correlated predictors (AR(1), rho = 0.5)
- Only 5 coefficients are truly nonzero; noise sd = 3
- Everything is simulated, so every method is graded against the truth

## Key findings

*(Fill in after running: best ridge/lasso test error versus OLS, the lambda
that minimised it, and lasso precision and recall.)*

## What is in it

- **Ridge**: closed form, and the SVD view in which each OLS component is
  multiplied by d² / (d² + nλ)
- **Hoerl–Kennard theorem**: the ridge error has negative slope at λ = 0, so
  some λ > 0 always beats OLS
- **Lasso**: no closed form; solved exactly by coordinate descent with
  soft-thresholding, and checked against the KKT optimality conditions
- **Orthonormal design**: lasso equals soft-thresholded OLS, ridge equals
  OLS / (1 + λ)
- **Bias-variance decomposition**: measured by redrawing noise on a fixed
  design and compared with exact ridge formulas
- **Cross-validation**: minimum and 1-standard-error rules, evaluated on
  support recovery (precision and recall)

## Tools

R, base linear algebra (`svd`, `solve`), dplyr, tidyr, tibble, ggplot2, Quarto.
No modelling packages: ridge, lasso and CV are written from scratch.

## Reproduce

1. `install.packages(c("dplyr", "tidyr", "tibble", "ggplot2", "knitr", "quarto"))`
2. Run `R/01_run_experiments.R` (saves `data/clean/verification_results.rds`)
3. Render `reports/report.qmd`

No data download is needed.

## Project structure

​```
R/regularization_functions.R   ridge, lasso (coordinate descent), KKT check, CV, exact ridge bias/variance
R/01_run_experiments.R         theory checks, bias-variance simulation, estimator comparison
reports/report.qmd             proofs, results and interpretation
data/clean/                    saved results (generated)
​```

## Limitations

- One simulation design; dense truths favour ridge and stronger correlation
  hurts lasso selection, neither of which is varied here
- Predictors are standardised once before cross-validation, so held-out folds
  slightly influence the scaling
- Teaching-grade implementation; use `glmnet` for real work
- Results carry Monte Carlo noise from a finite number of replications
- Consistency of lasso variable selection and guarantees for CV-chosen lambda
  are not proved
