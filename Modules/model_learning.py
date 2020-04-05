import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from datetime import timedelta
import time
import pickle
import os


def balancing(X, y):
    X, y = SMOTE(random_state=42, sampling_strategy=0.3).fit_sample(X, y)
    return X, y


def data_and_target(train):
    X_train = train.drop(['user_id', 'is_churned'], axis=1)
    y_train = train['is_churned']
    return X_train, y_train


def xgb_fit(X_train, y_train):
    clf = xgb.XGBClassifier(max_depth=3,
                            n_estimators=100,
                            learning_rate=0.1,
                            nthread=5,
                            subsample=1.,
                            colsample_bytree=0.5,
                            min_child_weight=3,
                            reg_alpha=0.,
                            reg_lambda=0.,
                            seed=42,
                            missing=1e10)

    clf.fit(X_train, y_train, eval_metric='aucpr')
    return clf


if __name__ == '__main__':
    start_t = time.time()
    path = os.path.dirname(os.path.abspath(__file__))
    print(f'Train data loading')
    train = pd.read_csv(os.path.join(path, 'dataset', 'dataset_train.csv'), sep=';')

    # Обучение и сохранение моделей масштабирования и XGBoost в файлы pickle
    X, y = data_and_target(train)

    # Масштабирование
    print(f'Data scaling')
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    if not os.path.exists('models'):
        os.makedirs('models')
    print('Run_time (data loading and scaling):', str(timedelta(seconds=(time.time() - start_t))))
    pickle.dump(scaler, open(os.path.join(path, 'models', 'scaler.pcl'), 'wb'))

    # Балансировка, обучение модели и её сохранение
    print('SMOTE balancing')
    X_balanced, y_balanced = balancing(X_scaled, y)
    print('Run_time (balancing):', str(timedelta(seconds=(time.time() - start_t))))

    print('XGBoost model training')
    model = xgb_fit(X_balanced, y_balanced)
    print('Run_time (training):', str(timedelta(seconds=(time.time() - start_t))))

    pickle.dump(model, open(os.path.join(path, 'models', 'xgboost.pcl'), 'wb'))
    print('Total time:', str(timedelta(seconds=(time.time() - start_t))))
