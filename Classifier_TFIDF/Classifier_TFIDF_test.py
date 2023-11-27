import os
import shutil
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def clear_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


# define file path
model_folder = "./classification_model"
test_data_path = "./y_X_cate/test_y_X_cate.csv"
save_y_pred_path = './classification_y_pred'

clear_dir(save_y_pred_path)

data = np.genfromtxt(test_data_path, delimiter=',')
X_test = data[:, 1:]
y_test = data[:, 0]
print("Finished Reading Test Data")

for model_file in os.listdir(model_folder):
    if model_file.endswith('.joblib'):
        model_name = model_file.replace('.joblib', '')
        # Load the model
        model_path = os.path.join(model_folder, model_file)
        model = joblib.load(model_path)
        print(f'Loaded {model_file}')

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Save predictions to CSV
        np.savetxt(f'{save_y_pred_path}/{model_name}_y_pred.csv', y_pred, delimiter=',')

        # Output the information
        print(f'Model: {model_name}')
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-Score: {f1}')
        print('-----------------------------------')
