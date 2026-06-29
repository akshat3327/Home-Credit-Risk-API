import joblib
import shap
import numpy as np

from src.feature_engineering import engineer_features
from src.data_pipeline.preprocess import preprocess


xgb_model = joblib.load("models/saved_models/xgb_model.pkl")
cat_model = joblib.load("models/saved_models/cat_model.pkl")

preprocessor = joblib.load("models/saved_models/preprocessor.pkl")

explainer = shap.TreeExplainer(xgb_model)


def predict(df, threshold):

  
    engineered_df = engineer_features(df)

  
    expected_columns = preprocessor.feature_names_in_

    for column in expected_columns:
        if column not in engineered_df.columns:
            engineered_df[column] = np.nan

 
    engineered_df = engineered_df[expected_columns]
    print("Expected columns:", len(expected_columns))
    print("Engineered columns:", len(engineered_df.columns))
    print("Missing:", set(expected_columns) - set(engineered_df.columns))

   
    processed_df = preprocess(engineered_df)

  
    xgb_prob = xgb_model.predict_proba(processed_df)[:, 1]
    cat_prob = cat_model.predict_proba(processed_df)[:, 1]

   
    final_probability = 0.5 * xgb_prob + 0.5 * cat_prob

   
    decision = (
        "Reject"
        if final_probability[0] >= threshold
        else "Approve"
    )

  
    shap_values = explainer.shap_values(processed_df)

    sample_shap = shap_values[0]

    feature_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in preprocessor.get_feature_names_out()
    ]

    feature_impacts = []

    for feature, impact in zip(feature_names, sample_shap):

        feature_impacts.append(
            {
                "feature": feature,
                "impact": float(impact)
            }
        )

    feature_impacts.sort(
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    top_features = feature_impacts[:5]

    
    return {

        "probability": float(final_probability[0]),

        "decision": decision,

        "top_features": top_features

    }