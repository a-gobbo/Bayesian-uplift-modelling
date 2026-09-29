# Bayesian Causal Inference & Uplift Modeling (MCMC)

This repository contains an end-to-end Data Science project demonstrating how to use **Bayesian Inference (PyMC & NUTS)** to perform causal uplift modeling for marketing promotions. 

Traditional predictive models ($P(Y=1 \mid X)$) often conflate a customer's organic baseline predisposition to buy with the actual incremental impact of a campaign. This project uses **Bayesian Logistic Regression** and **G-Computation** to isolate the true causal uplift of two different promotions (BOGO vs. Discount) compared to a No-Offer baseline.

## 🚀 Key Highlights
- **Causal Inference**: Transition from correlation to causation by adjusting for 9 key confounders (Recency, Monetary History, Channels, etc.).
- **Bayesian Uncertainty**: Instead of single-point estimates (p-values), the model computes the exact **Posterior Probability Distributions** of the promotional uplift.
- **Downside Risk & ROI Analysis**: Converts probabilistic distributions into actionable business metrics:
  - 95% Highest Density Intervals (HDI).
  - 5th-percentile Worst-Case Scenario (Value-at-Risk).
  - Unit economics simulation factoring in RFM-based Average Order Value (AOV), promotional cannibalization ("Sure Things"), and dispatch costs.

## 📂 Repository Structure

The workflow is divided into three sequential Jupyter Notebooks:

### 1. `01_data_preprocessing.ipynb`
Data ingestion, exploratory checks, categorical one-hot encoding, $\log(x+1)$ transformations for heavy-tailed monetary variables, and rigorous Z-score standardization. Outputs the clean dataset ready for PyTensor.

### 2. `02_bayesian_mcmc_model.ipynb`
Implementation of the Bayesian Generalized Linear Model (GLM) using **PyMC**.
- Specifies weakly informative regularizing priors ($\mathcal{N}(0, 1.0)$) to prevent overfitting.
- Utilizes the **No-U-Turn Sampler (NUTS)** for efficient Hamiltonian Monte Carlo exploration.
- Includes thorough convergence diagnostics: Gelman-Rubin $\hat{R}$, Effective Sample Size (ESS), and Traceplots.

### 3. `03_uplift_and_decision_making.ipynb`
The decision-theoretic culmination of the project.
- Implements **Robins' G-Computation** over the posterior trace to marginalize out confounders and compute population-level counterfactuals.
- Generates beautiful KDE distributions of causal uplift.
- Simulates a 100,000-customer marketing rollout, accounting for margin destruction (cannibalization) and computing the exact probability of realizing a negative ROI.

## 🛠️ Environment & Setup

To reproduce the analysis locally:
1. Ensure you have Python 3.9+ installed.
2. Install the required dependencies: `pip install pymc arviz pandas numpy matplotlib seaborn scipy`
3. *macOS Users*: PyTensor requires a native C++ compiler for optimization. If you encounter `<iostream>` missing errors, use the provided `run_clang_wrapper.sh` to correctly link your Apple Command Line Tools SDKs.

## 📊 Results Summary
The Bayesian model rigorously demonstrates that the **Discount** strategy is strictly superior to the **BOGO** strategy, delivering an expected net incremental profit of **~$50,000** per 100,000 users, with a risk of loss tightly bounded at `<6%`.
