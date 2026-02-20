# Feature Training–Serving Skew Fix

## Context
As the project evolved, feature preparation logic existed separately in the training pipeline and the API inference pipeline. Both paths cleaned and transformed data independently.

## Problem
After introducing a centralized feature builder, training failed because the target column **"Churn Value"** was accidentally included inside the feature matrix `X`.

Even though the target was extracted using:

y = df["Churn Value"]

the column still remained inside the dataframe.

## Root Cause
In pandas, selecting a column does not remove it from the dataframe.  
The feature builder removed the target only during inference but not during training.

This caused:

- model training with label leakage
- mismatch between training and inference schemas
- ColumnTransformer failures

## Solution
The feature builder contract was redesigned:

- target column is always removed
- feature builder outputs only model inputs
- label extraction handled separately in training

## Result
- identical feature schema in training and inference
- pipeline stability restored
- prevention of training–serving skew

## Key Lesson
If a feature cannot exist at inference time, it must never be used during training.
