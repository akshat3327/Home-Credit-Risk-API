import joblib


preprocessor = joblib.load("models/saved_models/preprocessor.pkl")


def preprocess(df):
    
    return preprocessor.transform(df)