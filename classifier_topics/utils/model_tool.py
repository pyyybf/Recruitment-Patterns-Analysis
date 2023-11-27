import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from output_tool import output


def prepare_data(file_path, topic_num=20, classifier=False):
    df = pd.read_csv(file_path)

    X = df[[f"Topic {i}" for i in range(0, topic_num)]]
    y = df["change_rate"]
    if classifier:
        y = (y > 0).astype(int)

    return X, y


def evaluate_model(y_pred, y_test, output_path=None):
    conf_mat = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f_measure = f1_score(y_test, y_pred)

    output("Confusion Matrix:\n", conf_mat, output_path=output_path)
    output("Accuracy:", accuracy, output_path=output_path)
    output("Precision:", precision, output_path=output_path)
    output("Recall:", recall, output_path=output_path)
    output("F1 Score:", f_measure, output_path=output_path)
