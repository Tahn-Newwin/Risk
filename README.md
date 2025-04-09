# 🧾 Credit Default Risk Analysis using Bayesian Inference

## 📌 Objective
Estimate the probability of loan default for applicants with missing features using Bayesian inference. This approach allows informed decision-making even with incomplete data.

## 📂 Dataset Overview
- **Size**: ~32,600 rows
- **Target**: `loan_status` (0 = non-default, 1 = default)
- **Key Features**: 
  - `person_age`
  - `person_income`
  - `person_emp_length`
  - `loan_amnt`
  - `loan_int_rate`
  - `loan_percent_income`
  - `cb_person_default_on_file`
  - `person_home_ownership`
  - `loan_intent`
  - `loan_grade`
  - `cb_person_cred_hist_length`

## 🧹 Preprocessing
- Removed rows with missing values (stored separately for prediction).
- Grouped continuous variables into bins based on median or meaningful thresholds.

## 🧠 Model Approach: Bayesian Updating
1. **Prior Probability**:
   - Calculated as the overall default rate in cleaned data.
2. **Likelihood**:
   - For each feature in a user profile, retrieve \( P(\text{feature value} | \text{default}) \).
   - Multiply all feature likelihoods.
3. **Posterior Probability**:
   - Combine prior and likelihood to compute default probability.
   - Skips any feature that is missing or unseen.

## 📖 Data Storytelling

### 🔍 What We Wanted to Know
> *"Can we still estimate how risky a borrower is, even if some of their data is missing?"*

### 🧩 What We Did
We used **Bayesian inference** to simulate how a loan officer might think:
> “Given what we *do* know about this person, how likely are they to default?”

For each user profile:
- We looked at known features (e.g., loan grade, income group).
- We **updated our belief** (prior) about default risk using what’s known.

### 🧠 What the Model "Believes"
- If a borrower’s known features **frequently appear in past defaults**, the model increases the risk score.
- If features are more common in **non-default cases**, the model becomes more optimistic.

### 📌 Real Example
A user had the following profile:
- Low income
- High interest rate
- Loan grade 'G'
- Short employment

These traits **heavily skewed toward defaults** in the past. As a result, the model predicted a **default chance >90%**, even though some data was missing.

This isn't just math — it's the model reflecting how **high-risk patterns** behave historically.

### 🧠 What It Means for Risk Management
- **Partial data ≠ no insight**: We can still make informed estimates.
- **Some features carry more "signal"** than others — loan grade and interest rate were particularly influential.
- **Extreme values cause high confidence** — when a combination of features appears only in default cases, the model reflects that decisively.

### 🚦 Actionable Takeaways
- This method gives a **risk score with uncertainty**, useful for human decision-makers.
- Instead of rejecting incomplete applications, lenders can **triage cases**:
  - ✅ Low-risk: fast-track approval
  - ❓ Medium-risk: ask for more info
  - ❌ High-risk: flag for review

## 📊 Sample Visualization Output
- Prior vs. Posterior distribution plot for each prediction.
- Highlighted user-specific probability of default.

## ⚠️ Assumptions
- All features are conditionally independent (naive Bayes assumption).
- Missing features are skipped during likelihood computation.
- Feature probabilities are based on historical default distributions only.
- No smoothing applied (no epsilon); unseen feature combinations may yield extreme probabilities.

## 📁 Output
- `filtered_missing_values.csv`: Stores incomplete rows.
- Console: Displays prediction and risk visualization.

## 🔄 Future Enhancements
- Add Laplace smoothing to avoid 0-probability issues.
- Incorporate a weighting scheme for more influential features.
- Use ML classifiers to compare performance with Bayesian model.
- Automate report generation with LLM to explain each prediction.

---
