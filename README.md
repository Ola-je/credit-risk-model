## Credit Scoring Business Understanding

In the context of building a credit risk probability model for Bati Bank, understanding the underlying business needs, regulatory environment, and data challenges is paramount. This section summarizes key aspects of credit scoring from a business perspective.

### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord is an international banking regulation that sets standards for capital adequacy, focusing on three pillars: minimum capital requirements, supervisory review, and market discipline. Its strong emphasis on robust risk measurement and management directly influences our model development in several ways:

* **Transparency and Explainability:** Basel II requires banks to hold sufficient capital against risks. To justify these capital allocations and for internal risk management, models must not be "black boxes." Regulators and internal auditors need to understand *how* the model arrives at its decisions, *which factors* contribute most to a risk score, and *why* a particular applicant is classified as high or low risk. An interpretable model (like Logistic Regression, especially when paired with WoE) allows for clear explanations of risk drivers.
* **Validation and Auditability:** The model must be regularly validated and audited. This necessitates clear documentation of the model's methodology, assumptions, data sources, and performance metrics. A well-documented model is easier to validate against regulatory standards and internal policies.
* **Capital Allocation:** The outputs of the credit risk model directly impact the bank's regulatory capital requirements. If the model is not sufficiently robust, interpretable, or documented, it may not be approved for internal ratings-based (IRB) approaches, forcing the bank to use less favorable standardized approaches.
* **Fairness and Compliance:** An interpretable model helps ensure fairness and prevent discriminatory lending practices, aligning with broader regulatory expectations beyond just financial stability.

### Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

Our project's core challenge is the absence of a direct "default" label in the eCommerce transaction data. In traditional credit scoring, historical loan performance (e.g., missed payments, bankruptcies) directly defines default. To apply supervised machine learning techniques, which require labeled data, a proxy variable is essential.

* **Necessity of a Proxy:**
    * **Supervised Learning:** Machine learning algorithms learn patterns from labeled data. Without a clear "default" (bad) or "non-default" (good) label, we cannot train a model to predict actual credit risk.
    * **Leveraging Alternative Data:** By defining a proxy based on behavioral patterns (Recency, Frequency, Monetary - RFM) derived from transactional data, we can transform an unlabeled dataset into one suitable for supervised learning, unlocking the predictive power of alternative data. The assumption is that customers who exhibit "disengaged" or "low-value" behavior are a reasonable proxy for higher credit risk.

* **Potential Business Risks of Using a Proxy:**
    * **Misclassification Risk:** The primary risk is that the proxy may not perfectly align with actual credit default. A customer labeled "high-risk" by our proxy might, in reality, be creditworthy and repay a loan. This could lead to:
        * **Lost Revenue:** Denying loans to creditworthy customers means missed interest income and potential customer churn for the BNPL service.
        * **Reduced Market Penetration:** An overly conservative proxy could limit the bank's ability to onboard new customers for the BNPL service.
    * **Suboptimal Model Performance:** If the proxy has a weak correlation with true default behavior, the model's predictions will be less accurate, leading to inefficient capital allocation or higher unexpected losses.
    * **Bias Introduction:** The definition of "disengaged" customers might inadvertently introduce biases if certain demographic or behavioral patterns are unfairly penalized, potentially leading to discriminatory lending outcomes. This requires careful analysis and validation of the proxy's fairness.

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

The choice of model type involves balancing predictive performance with business and regulatory requirements.

* **Simple, Interpretable Model (e.g., Logistic Regression with WoE):**
    * **Pros:**
        * **High Interpretability:** Coefficients are easily understood; the impact of each feature on the probability of default is clear. This directly facilitates compliance with Basel II's interpretability requirements.
        * **Transparency:** Easier to explain to non-technical stakeholders, regulators, and customers.
        * **Ease of Validation:** Simpler to audit, test assumptions, and validate against historical data or expert judgment.
        * **Robustness:** Often less prone to overfitting with proper feature engineering (like WoE, which linearizes relationships on a log-odds scale).
        * **Traditional Acceptability:** Logistic Regression with WoE is a long-standing, widely accepted method in credit scoring and scorecard development.
    * **Cons:**
        * **Potentially Lower Performance:** May not capture highly complex, non-linear relationships or interactions between features as effectively as more advanced models, potentially leading to slightly lower predictive accuracy.
        * **Feature Engineering Intensive:** Requires significant upfront feature engineering (e.g., binning, WoE transformation) to perform well.

* **Complex, High-Performance Model (e.g., Gradient Boosting Machines):**
    * **Pros:**
        * **Higher Predictive Accuracy:** Often achieves superior performance by modeling intricate, non-linear relationships and complex feature interactions.
        * **Handles Data Automatically:** Can often handle missing values and outliers more robustly with less explicit preprocessing.
    * **Cons:**
        * **"Black Box" Nature:** Often difficult to interpret *why* a specific prediction was made, making it challenging to explain to regulators and justify decisions. This is a significant hurdle in regulated financial environments.
        * **Complex Validation:** More difficult to validate and audit due to their opaque nature.
        * **Higher Overfitting Risk:** More prone to overfitting if not carefully tuned, especially with smaller datasets.
        * **Less Regulatory Acceptance (Historically):** While gaining traction, justifying complex models can still be harder in conservative financial institutions compared to traditional scorecard models. (However, techniques like SHAP or LIME are helping to bridge this gap).

* **Overall Trade-off:** In a highly regulated industry like finance, **interpretability and explainability often take precedence over marginal gains in raw predictive accuracy.** A model that is easily understood, validated, and documented is generally preferred to meet compliance requirements and build trust, even if a more complex model might offer slightly better performance. The choice ultimately depends on regulatory appetite, available interpretability tools, and the specific risk appetite of Bati Bank.