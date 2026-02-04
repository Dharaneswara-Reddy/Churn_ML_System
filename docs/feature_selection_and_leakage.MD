# Feature Selection and Leakage Prevention

This document explains why certain columns were explicitly dropped while building the churn prediction model.  
The goal is to prevent data leakage and ensure that the model behaves realistically in production.

---

## Core Principle

A model may only use information that exists at prediction time.

If a feature:
- is calculated after churn happens,
- is derived from churn logic,
- or embeds human or model judgment about churn,

then it should be treated as **leakage**, not **insight**.

Data leakage leads to:
- inflated accuracy during training,
- false confidence in model performance,
- and models that fail when deployed in the real world.

For this reason, several columns were intentionally excluded from the feature set.

---

## Dropped Columns and Rationale

### 1. CustomerID and Count

`CustomerID` is a unique identifier and does not carry any predictive signal.  
Including it would only add noise and potentially cause the model to memorize specific records.

`Count` is a reporting/dashboarding column used for aggregation and has no relevance to individual churn behavior.

Both columns were dropped as they do not contribute meaningful information to the prediction task.

---

### 2. Churn Label

`Churn Label` is simply the string representation (Yes/No) of `Churn Value` (1/0).

Including it would directly expose the target to the model, making training meaningless and invalid.  
For this reason, it was dropped and `Churn Value` was used as the numeric target instead.

---

### 3. Churn Score

`Churn Score` is a value between 0 and 100 calculated using IBM SPSS Modeler.

This column represents the output of another churn prediction model.  
If used as a feature, the model would effectively learn from an existing prediction rather than from raw customer behavior.

This would result in:
- unrealistically high accuracy during training,
- a model that learns nothing useful,
- and failure in production where this score would not be available.

Therefore, `Churn Score` was excluded to avoid model-on-model leakage.

---

### 4. Churn Reason

At first glance, `Churn Reason` appears to provide strong insight.

However, the reason for churn is only known **after** the customer has already left.  
Using this column would inject future information into the model, which violates causality.

For this reason, `Churn Reason` was dropped from the feature set.  
That said, it remains valuable for post-churn analysis and business understanding, just not for prediction.

---

### 5. Churn Category

`Churn Category` is a higher-level abstraction derived from churn reasons.

Using it would be equivalent to predicting an event using its explanation.  
This again introduces leakage, as it depends on information that is only available after churn occurs.

Hence, it was excluded for the same causality reasons as `Churn Reason`.

---

### 6. CLTV (Customer Lifetime Value)

CLTV provides a lot of business insight, but it is not suitable as a model feature.

It is typically a predicted or calculated future value that often includes assumptions about:
- tenure,
- revenue,
- and churn behavior itself.

Including CLTV would leak future expectations into the model and create circular logic, leading to unreliable predictions in production.

CLTV is better used **after** churn probability is predicted, for prioritization and decision-making, rather than as an input feature.

---

## Summary

All dropped columns were excluded for one core reason:  
they either reveal the target directly, depend on future information, or embed prior churn knowledge.

By removing these columns, the model is forced to learn from genuine customer behavior that would actually be available at prediction time, resulting in a more realistic and production-ready churn prediction system.
