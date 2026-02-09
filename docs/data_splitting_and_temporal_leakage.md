# Data Splitting Strategy and Temporal Leakage

Initially, the dataset was split into training and testing sets using a random train-test split. This approach seemed reasonable because it is commonly used in machine learning workflows and ensures that both sets have a similar distribution of data. At first glance, this gave stable and relatively good metrics, which made the model appear reliable.

---

## The Problem with Random Splits 

The problem with a random split is that it mixes past and future data together. In the context of churn prediction, this means the model may see patterns from customers who churned later in their lifecycle while training, and then be evaluated on customers from earlier stages. This creates temporal leakage, where future information indirectly influences training. As a result, the model appears to perform well, but the metrics are overly optimistic and do not reflect how the model behave in a real production scenario.

--- 

## Temporal Signal in the Dataset

Although the dataset does not contain an explicit timestamp, it includes the `Tenure Months` feature, which strongly represents time. Customer behavior changes as tenure increases, and churn patterns for new customers are different from those of long-term customers. Ignoring this temporal signal assumes that customer behavior is stationary over time, which is not true for churn problems. Treating all tenures as interchangeable hides this shift and introduces leakage.

---

## Change in Splitting Strategy

To address this issue, the splitting strategy was changed to a time-aware split using `Tenure Months` as a proxy for time. The data was sorted by tenure, with earlier-tenure customers used for training and later-tenure customers reserved for testing. This resulted in a clear separation, where the training set covered lower tenure ranges and the test set covered higher tenure ranges. This approach better simulates how the model would be deployed in practice, where predictions are made on future customers using historical data.

--- 

## Observed Impact on Metrics

After switching to a time-aware split, both precision and recall dropped significantly, and threshold-based tuning became less effective. This happened because the model was no longer evaluated on data similar to what it had already seen during training. Instead, it was forced to generalize to new customer behavior patterns. While this drop may seem concerning at first, it is expected and correct, as it reflects the true difficulty of predicting churn under realistic conditions rather than benefiting from leaked information.

--- 

## Production Implications 

This experiment highlights that churn behavior is non-stationary and changes over time. A static model trained once on historical data will gradually lose effectiveness as customer behavior evolves. Without proper handling, model performance can silently degrade in production. This implies that real-world churn systems require ongoing monitoring, retraining strategies, and possibly segmentation based on customer lifecycle stages to remain reliable.

--- 

## Summarization 

Random train-test splits can hide temporal leakage and create misleading confidence in model performance. By introducing a time-aware split, the system exposed realistic performance limitations and revealed the true challenges of churn prediction. This trade-off favors honesty and production realism over inflated offline metrics.
