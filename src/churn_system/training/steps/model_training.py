"""
Multi Model Training 

Trains Multiple candidate models and selects the best performing model 
automatically (Champion Selection)
"""

import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from churn_system.logging.logger import get_logger
from churn_system.config.config import CONFIG

logger = get_logger(__name__, CONFIG["logging"]["training"])

def build_preprocessor(X:pd.DataFrame):
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    
    return preprocessor

def get_candidate_models():
    return{
        "logistic_regression" : LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "random_forest" : RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "gradient_boosting" : GradientBoostingClassifier()
    }

def evaluate(model,X_test,y_test):
    
    probs = model.predict_proba(X_test)[:,1]
    preds = (probs >= 0.5).astype(int)
    
    metrics = {
        "accuracy" : float(accuracy_score(y_test,preds)),
        "precision" : float(precision_score(y_test,preds)),
        "recall" : float(recall_score(y_test,preds)),
        "f1_score" : float(f1_score(y_test,preds)),
        "roc_auc" : float(roc_auc_score(y_test, probs)),
        "pr_auc" : float(average_precision_score(y_test, probs)),
    }
    
    return metrics

def train_model(X_train, y_train, X_test, y_test):
    """
    Train Multiple models and return the best one.
    """

    logger.info("Starting multi-model training.")

    preprocessor = build_preprocessor(X_train)
    candidates = get_candidate_models()
    
    best_model = None
    best_metrics = None
    best_score = -1
    best_name = None
    
    for name, model in candidates.items():
        logger.info(f"Training Candidate model : {name}")
        pipeline = Pipeline(
            steps = [
                ("preprocessor" , preprocessor),
                ("model", model),
            ]
        )
        
        pipeline.fit(X_train,y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        logger.info(f"{name} ROC-AUC = {metrics['roc_auc']:.4f}")
        
        # Champion Selection based on ROC_AUC
        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = pipeline
            best_metrics = metrics
            best_name = name
            
    logger.info(f"Champion model selected: {best_name}")
    return best_model, best_metrics, best_name