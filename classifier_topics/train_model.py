import os
import joblib

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from utils.model_tool import prepare_data, evaluate_model
from classifier_topics.utils.output_tool import output
from utils import paths


def train_model(model_name, X_train, y_train):
    if model_name == "LogisticRegression":
        param_grid = {
            "penalty": ["l1", "l2"],
            "C": [0.01, 0.1, 1, 10, 100],
        }
        clf = LogisticRegression(solver="liblinear")
    elif model_name == "KNeighborsClassifier":
        param_grid = {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "p": [1, 2]
        }
        clf = KNeighborsClassifier()
    elif model_name == "SVC":
        param_grid = {
            "C": [0.01, 0.1, 1, 10, 100],
            "kernel": ["linear", "rbf"],
        }
        clf = SVC(gamma="auto")
    elif model_name == "DecisionTreeClassifier":
        param_grid = {
            "max_depth": [None, 3, 5, 8, 15, 25, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        clf = DecisionTreeClassifier()
    elif model_name == "RandomForestClassifier":
        param_grid = {
            "n_estimators": [50, 100, 300, 500, 800],
            "max_depth": [None, 5, 15, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        clf = RandomForestClassifier()
    elif model_name == "GradientBoostingClassifier":
        param_grid = {
            "n_estimators": [100, 300, 500, 800, 1200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10],
        }
        clf = GradientBoostingClassifier()
    elif model_name == "NaiveBayes":
        param_grid = {}
        clf = GaussianNB()
    elif model_name == "NeuralNetworks":
        param_grid = {
            "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)],
            "alpha": [0.0001, 0.001, 0.01]
        }
        clf = MLPClassifier()
    else:
        param_grid = {}
        clf = None

    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    grid_search = GridSearchCV(clf, param_grid, cv=stratified_kfold, scoring="accuracy", verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


if __name__ == "__main__":
    # Create the directory for saving models
    if not os.path.exists(paths.save_model_dir):
        os.makedirs(paths.save_model_dir)
    # Clear output file
    with open(paths.output_path, "w"):
        pass

    # Read train & test dataset
    X_train, y_train = prepare_data(paths.train_data_path, classifier=True)
    X_test, y_test = prepare_data(paths.test_data_path, classifier=True)

    # Normalize features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier_names = [
        "LogisticRegression",
        "KNeighborsClassifier",
        "SVC",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "NaiveBayes",  # GaussianNB
        "NeuralNetworks",  # MLPClassifier
    ]
    for classifier_name in classifier_names:
        output(f"\n========== {classifier_name} ==========", output_path=paths.output_path)
        # Train model
        model = train_model(classifier_name, X_train, y_train)
        output(model, output_path=paths.output_path)
        # Test model
        y_pred = model.predict(X_test)
        evaluate_model(y_pred, y_test, output_path=paths.output_path)
        # Save model
        joblib.dump(model, f"{paths.save_model_dir}/{classifier_name}.joblib")
