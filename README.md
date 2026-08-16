# The $99k Pivot: Applying Dynamic Microeconomics to Machine Learning Fraud Detection

In the standard application of predictive data science, success is usually measured by a single classification metric—like an F1-Score or a Precision-Recall curve. But algorithms do not pay the bills; businesses do. By replacing a standard machine learning threshold with a microeconomic cost function, I transitioned a static fraud detection model into a profit-maximizing engine, rescuing 1,134 legitimate customer relationships and generating **$99,136.74 in ultimate net savings**.

**The Core Thesis**
Machine learning models are brilliant at calculating probability, but they are historically terrible at calculating business value. Out of the box, standard classification algorithms blindly assume that every error is equally damaging. They assume that falsely declining a customer’s $4 coffee purchase carries the exact same operational penalty as allowing a $5,000 fraudulent wire transfer to slip through. 

In the real world of financial systems, costs are highly asymmetrical. Customer churn has a lifetime value (LTV) penalty, and catastrophic fraud carries chargeback fines. To bridge the gap between predictive accuracy and actual corporate profit, an algorithm must be forced to abandon the naive "0.50 cutoff" and adopt a dynamic, microeconomic threshold.

---

##  The Data Source and Local Feature Classification

**The Dataset**
The foundational data for this microeconomic simulation was sourced from the **IEEE-CIS Fraud Detection dataset**, originally provided by Vesta Corporation. The dataset contains hundreds of thousands of real-world e-commerce transactions, providing a highly imbalanced, structurally realistic environment to test econometric theories. After standard preprocessing and a rigorous train/test split, the final hold-out validation set evaluated in this study contained roughly 85,000 legitimate transactions and a proportionate subset of confirmed frauds.

**Local Data Classification & PII Governance**
In live financial environments, data must be strictly governed and locally classified to protect consumer privacy. To mirror corporate compliance standards, the features in this dataset were anonymized and locally classified into distinct categorical risk buckets:

*   **Transactional Features (`TransactionAmt`, `ProductCD`):** These variables capture the direct economic weight and the specific product category of the purchase. They form the core of the dynamic LTV friction equation, acting as the primary scalars for financial risk.
*   **Card & Banking Classifications (`card1` - `card6`):** To protect Personally Identifiable Information (PII), actual credit card numbers were hashed and locally classified into categorical buckets (e.g., card type, issuing bank, network). This allows the algorithm to detect systemic risks across specific banking networks without exposing individual user identities.
*   **Temporal & Velocity Metrics:** Rather than using raw timestamps, temporal data was classified into velocity metrics to track anomalous user behavior, forming the primary machine-learning defense against high-speed bot attacks.

---

##  Econometric Feature Engineering

To capture true consumer behavior, the raw data required three specific mathematical transformations:

1. **Cyclical Time:** Temporal data was encoded using sine and cosine functions to capture the continuous loop of time without artificial mathematical breaks at midnight.
2. **Velocity Metrics:** Rolling time-series features were built to track transaction frequency, allowing the model to instantly flag high-speed, automated bot attacks and "card testing."
3. **Centered Variables:** Continuous variables were represented strictly as the difference from the sample mean to eliminate structural multicollinearity (making polynomial and interaction terms orthogonal) and stabilize the model's coefficients. 

Together, these engineered features successfully transformed static data points into dynamic, highly predictive behavioral signals.

---

##  The Flaw of the 0.50 Threshold: An Assumption of Symmetrical Costs

When a standard predictive model is deployed out of the box, it relies on a rigid decision boundary: the 0.50 threshold. If the algorithm calculates that a transaction has a 51% probability of being fraudulent, it blocks the card. If the probability is 49%, it approves the transaction. 

Mathematically, this threshold relies on a fatal flaw known as the **assumption of symmetrical costs**. It assumes that the penalty for a False Positive (falsely blocking a legitimate customer) is exactly equal to the penalty for a False Negative (allowing a fraudster to successfully steal). Standard algorithms treat every error as a simple "-1" to their overall accuracy score.

In the operational reality of financial systems, costs are aggressively asymmetrical:
*   **The High-Dollar Miss (False Negative):** If a model flags a highly suspicious $5,000 international wire transfer, but calculates the probability at only 45%, the naive 0.50 threshold blindly approves it. The algorithm ignored the capital at risk, costing the bank $5,000 plus chargeback fines.
*   **The Low-Dollar Friction (False Positive):** Conversely, if the model is 51% sure that a $3 coffee purchase is fraud, it will strictly decline the card. The bank saves $3, but inflicts catastrophic friction on a legitimate customer, risking the loss of their Lifetime Value (LTV) to a competitor over pocket change.

To measure this failure, I ran a highly calibrated XGBoost model through the hold-out test set using this standard 0.50 cutoff. Because the algorithm was entirely blind to the dollar amounts at stake, it generated a baseline expected financial loss of **$422,326.43**. 

The model wasn't failing to detect data patterns; it was failing to understand economics. 

---

##  Bridging ML and Econometrics: The Dynamic BMR Threshold

To fix the structural blindness of the baseline model, the algorithm required a microeconomic cost function. Instead of asking the model, "Is this transaction fraudulent?", I engineered the system to calculate, "What is the financial risk of this classification?"

This requires replacing the static probability cutoff with a dynamic **Bayes Minimum Risk (BMR)** framework. The BMR model calculates a unique decision threshold, $p^*$, for every individual transaction based on its specific financial weight:

$$p^* = \frac{C_{FP}}{C_{FP} + C_{FN} - C_{TP}}$$

In this equation, $C_{FP}$ represents the operational cost of a False Positive, $C_{FN}$ represents the cost of a False Negative (the total transaction amount lost to fraud plus a $20 chargeback fine), and $C_{TP}$ represents the minor administrative cost of a True Positive.

**The Dynamic LTV Proxy**
Traditional academic applications of BMR often rely on static administrative costs, assuming every False Positive costs a flat fee. From an econometric perspective, this is a massive oversimplification. The true cost of a False Positive is the risk of **customer churn**. 

To capture this elasticity, I programmed the $C_{FP}$ variable as a dynamic proxy for Customer Lifetime Value (LTV). The friction cost scaled strictly with the transaction size: a base penalty of $15.00 plus 15% of the transaction amount. A falsely declined $4 coffee carries negligible churn risk, but a falsely declined $4,000 international flight triggers catastrophic friction and a high probability of losing a high-net-worth account.

**The Economic Trade-Off**
When the model evaluated transactions using this dynamic $p^*$ threshold, it acted as a profit-maximizing engine. It dynamically lowered the threshold for high-dollar transactions to aggressively protect the bank's bottom line, while letting low-dollar, high-friction interventions slide.

The financial impact was immediate: the total expected loss dropped from the $422,326 baseline down to **$386,406**, generating roughly $36,000 in net savings. 

<img width="940" height="467" alt="image" src="https://github.com/user-attachments/assets/bb5ee61b-4df0-4969-8d0e-eba11617759c" />



As the visualization illustrates, the 2-Tier BMR system successfully secured the bottom line, but it achieved those savings by casting a much wider net. Catastrophic friction spiked dramatically, with legitimate customer blocks skyrocketing from 394 to **1,528**. The economic logic was sound, but operationally, absorbing that level of customer friction was unsustainable. 

---

##  The Reality Check: Why Econometrics Demands Calibrated Probabilities

The dynamic $p^*$ threshold is mathematically sound, but its success relies on a massive underlying assumption: **the algorithm must output the truth.**

When a model states that a transaction has a 4% probability of being fraud, the BMR equation takes that number literally to calculate financial risk. However, standard tree-based ensemble models (like XGBoost) are notoriously poor at outputting true probabilities. They are designed to optimize ranking, not reality, resulting in extreme overconfidence at the tails.

**Model Selection via Brier Scores**
To ensure the economic integrity of the system, I discarded standard classification metrics like AUC and F1-Score for model selection. Instead, I evaluated models using the **Brier Score**—a metric that strictly measures the mean squared difference between predicted probabilities and actual outcomes. XGBoost emerged as the most mathematically grounded model for this specific dataset. 

**Applying Isotonic Calibration**
To fix XGBoost’s natural overconfidence, I applied **Isotonic Regression** to calibrate the model against the hold-out validation set. Isotonic calibration acts as a mathematical reality check: it maps the model's distorted raw outputs back to the actual, empirical fraud rates observed in the data.

When the newly calibrated, highly accurate probabilities collided with the aggressive Dynamic LTV BMR threshold, the model stopped guessing and started calculating true risk. It was this exact collision that drove the expected financial loss down to $386,406—but it was also this collision that forced the model to intervene on 1,528 legitimate customers. 

---

##  Operationalizing the Math: The 3-Tier SMS System

To save the bank's bottom line without burning the customer base, I designed a **3-Tier Operational Framework** that introduces a low-cost "Grey Zone" intervention. Instead of forcing the model into a binary "Approve vs. Decline" corner, I integrated an SMS Two-Factor Authentication (2FA) tier. 

The logic operates on three tiers:
1.  **Tier 1: Silent Approval.** If the calibrated probability is below the dynamic $p^*$ risk threshold, the transaction is approved silently. Zero friction.
2.  **Tier 3: Strict Hard Decline.** If the probability exceeds the standard 0.50 threshold, the model is highly confident it is fraud. The transaction is instantly blocked. 
3.  **Tier 2: The SMS Grey Zone.** If the probability crosses the hyper-sensitive $p^*$ threshold but has not yet reached the 0.50 mark, the system pauses. Instead of triggering a catastrophic hard decline, the system pings an API to send an SMS verification code to the user's phone at a cost of just **$0.05**. 
<img width="940" height="399" alt="image" src="https://github.com/user-attachments/assets/34f137c1-9af1-46f4-bee9-ed15b221b11d" />



**Rescuing the Customer Base**
The operational impact of this 3-Tier system was massive. As the visualization shows, the system took the massive wall of 1,528 hard declines and safely shaved off the top 75%. **1,134 legitimate customers were rescued** from a catastrophic hard decline and redirected into a minor 5-cent friction check.

---

## The Final Impact: Bridging Data Science and Business Reality

The transition from a standard machine learning baseline to a dynamically calibrated, 3-Tier economic system yielded a transformative financial result. 

When the original, uncalibrated algorithm was allowed to blindly enforce its rigid 0.50 threshold, it generated an expected financial loss of **$422,326.43**. 

By enforcing Probability Calibration and replacing the static threshold with a microeconomic cost function (Dynamic LTV BMR), the system learned to calculate financial risk rather than raw probability. Finally, by introducing the $0.05 SMS "Grey Zone," the model safely neutralized the spike in customer friction. 

The final, fully optimized 3-Tier system drove the expected financial loss down to **$323,189.69**. 

This represents an **Ultimate Net Savings of $99,136.74**, achieved while simultaneously rescuing 1,134 legitimate customer relationships from catastrophic hard declines. 

**The Broader Takeaway**
The most critical lesson of this project is that algorithms do not exist in a vacuum. In the financial sector, a model that strictly optimizes for classification metrics like F1-Score or AUC is fundamentally detached from business reality. 

The future of automated decision-making lies at the intersection of predictive data science and applied microeconomics. When we force our algorithms to stop asking, *"Is this transaction fraudulent?"* and teach them to calculate, *"What is the economic risk of this decision?"*, machine learning stops being a rigid IT expense and becomes a dynamic, profit-maximizing engine.
