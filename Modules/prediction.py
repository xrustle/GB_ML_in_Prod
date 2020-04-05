import pandas as pd
from datetime import timedelta
import time
import pickle
import os


if __name__ == '__main__':
    start_t = time.time()
    path = os.path.dirname(os.path.abspath(__file__))
    print(f'Data and models loading...')
    test = pd.read_csv(os.path.join(path, 'dataset', 'dataset_test.csv'), sep=';')
    scaler = pickle.load(open(os.path.join(path, 'models', 'scaler.pcl'), 'rb'))
    model = pickle.load(open(os.path.join(path, 'models', 'xgboost.pcl'), 'rb'))

    test_scaled = scaler.transform(test.drop(['user_id'], axis=1))

    print(f'Prediction...')
    y_pred = model.predict(test_scaled)
    test['is_churned'] = y_pred
    test[['user_id', 'is_churned']].to_csv('batorov_predictions.csv', sep=';', index=None)
    print('Total time:', str(timedelta(seconds=(time.time() - start_t))))
