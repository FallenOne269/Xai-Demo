import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lime import lime_tabular

def main():
    # 1. Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    # 4. LIME Explanation
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True
    )

    # Choose one sample to explain
    sample_idx = 0
    sample = X_test[sample_idx].reshape(1, -1)
    predicted_class = model.predict(sample)[0]
    print(f"Predicted class for sample {sample_idx}: {class_names[predicted_class]}")

    # Generate explanation
    exp = explainer.explain_instance(
        data_row=X_test[sample_idx],
        predict_fn=model.predict_proba,
        num_features=2
    )
    exp.show_in_notebook(show_table=True)

def jls_extract_def(explainer, data_row, X_test, sample_idx, predict_fn, model, num_features):
    if __name__ == "__main__":
        main()
    exp = explainer.explain_instance(
        data_row=X_test[sample_idx],
        predict_fn=model.predict_proba,
        num_features=2)
        




