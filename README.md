# Bayesian Causal Inference & Uplift Modelling (MCMC)

This repository contains a Data Science project demonstrating the use of **Bayesian Inference** to perform causal uplift modelling for marketing promotions. 

Using the "Marketing Promotion Campaign Uplift Modelling" dataset (Kaggle, 64k customers), I analyzed a historical promotional campaign in which users had received one of three treatments (No Offer, Discount, Buy One Get One).

I developed a Bayesian Logistic Regression model (PyMC, NUTS/HMC sampler), adjusting for 9 observed confounders (RFM metrics, acquisition channel) to correct for observational bias. Using Robins' G-computation over the posterior trace, I estimated population-level counterfactuals to quantify the causal uplift of each intervention.

Beyond point estimates, I propagated posterior uncertainty from the MCMC trace into a financial simulator: for each posterior draw, expected profit was computed via unit economics, yielding a full distribution of outcomes per strategy from which I derived expected profit and downside risk (VaR, probability of net loss).

The analysis showed that while the Discount strategy produced a higher raw conversion uplift, BOGO performed better financially once risk was accounted for: BOGO delivered a $30k higher expected net profit per 100k users, with a <4% probability of net loss, and a positive 5th-percentile (VaR₉₅) outcome. Discount, by contrast, carried a $17k Value-at-Risk (VaR₉₅), exposing meaningful downside risk despite its higher point-estimate uplift.

##  Repository Structure

**Dataset Summary:** The project utilizes a marketing dataset comprising 64,000 customer records. Key variables include:
*   **Target:** `conversion` indicator, whether the customer bought the item.
*   **Treatment:** `offer` categorical assignment, evenly distributed across *Discount*, *Buy One Get One (BOGO)*, and *No Offer* (control group).
*   **Confounders:** Customer characteristics including historical spend (`history`, which is heavily right-skewed), months since last purchase (`recency`), geographic area (`zip_code`), referral status, and acquisition `channel`.

The workflow is divided into three sequential Jupyter Notebooks:

### 1. `01_data_preprocessing.ipynb`
Data ingestion, exploratory checks, $\log(x+1)$ transformations for heavy-tailed monetary variables, and Z-score standardization. Outputs the clean dataset ready for PyTensor.

### 2. `02_bayesian_mcmc_model.ipynb`
Implementation of the Bayesian Generalized Linear Model (GLM) using **PyMC**.
- Specifies weakly informative regularizing priors ($\mathcal{N}(0, 1.0)$) to prevent overfitting.
- Utilizes the **No-U-Turn Sampler (NUTS)** for efficient Hamiltonian Monte Carlo exploration.
- Includes thorough convergence diagnostics: Gelman-Rubin $\hat{R}$, Effective Sample Size (ESS), and Traceplots.

### 3. `03_uplift_and_decision_making.ipynb`
The culmination of the project.
- Implements **Robins' G-Computation** over the posterior trace to marginalize out confounders and compute population-level counterfactuals.
- Generates KDE distributions of causal uplift.
- Simulates a 100,000-customer marketing rollout, accounting for margin destruction (cannibalization) and computing the exact probability of realizing a negative ROI.

## Environment & Setup

To reproduce the analysis locally:
1. Ensure you have Python 3.9+ installed.
2. Install the required dependencies: `pip install pymc arviz pandas numpy matplotlib seaborn scipy`
3. *macOS Users*: PyTensor requires a native C++ compiler for optimization. If you encounter `<iostream>` missing errors, use the provided `run_clang_wrapper.sh` to correctly link your Apple Command Line Tools SDKs.

## Results Summary
The Bayesian model rigorously demonstrates that, although the discount vs BOGO uplift is **~3%**, the **BOGO** strategy is superior to the **discount** strategy, delivering an expected net incremental profit of **~$70,000** per 100,000 users, with a risk of loss tightly bounded at `<4%`.
