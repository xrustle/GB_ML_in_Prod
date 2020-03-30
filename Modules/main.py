from building_dataset import build_dataset_raw
from processing_dataset import prepare_dataset
from model_learning import x_and_y, scale, balancing, xgb_fit
from validation import split, evaluation
from prediction import xgb_predict
import pandas as pd
import pickle
import json
import os

path = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(path, 'config.json'), encoding='utf-8-sig') as json_data_file:
    conf = json.load(json_data_file)

START_DATE = conf['start']
END_DATE = conf['end']
INTER_LIST = conf['inter_list']


def build_dataset(mode):
    build_dataset_raw(churned_start_date=START_DATE,
                      churned_end_date=END_DATE,
                      inter_list=INTER_LIST,
                      raw_data_path=mode+'/',
                      dataset_path='dataset/',
                      mode=mode)


if __name__ == '__main__':
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    # Строим сырые датасеты
    build_dataset('train')
    build_dataset('test')

    # Обрабатываем датасеты, сохраняем
    prepare_dataset(dataset=pd.read_csv('dataset/dataset_raw_train.csv', sep=';'),
                    inter_list=INTER_LIST,
                    dataset_type='train')
    prepare_dataset(dataset=pd.read_csv('dataset/dataset_raw_test.csv', sep=';'),
                    inter_list=INTER_LIST,
                    dataset_type='test')

    train = pd.read_csv('dataset/dataset_train.csv', sep=';')
    test = pd.read_csv('dataset/dataset_test.csv', sep=';').drop(['user_id'], axis=1)

    # Разбитие на тренировочную и тестовую выборки. Метрики модели
    # X_train, y_train, X_test, y_test = split(train)

    # Обучение и сохранение модели в файл
    X, y = x_and_y(train)
    X, test = scale(X, test)
    model = xgb_fit(*balancing(X, y))
    pickle.dump(model, open(os.path.join(path, 'xgb.pickle'), 'wb'))

    # Предсказание ответов для тестовой выборки
    # model = pickle.load(open(os.path.join(path, 'xgb.pickle'), 'rb'))
    y_pred = xgb_predict(model, test)[1]
    y_pred_df = pd.DataFrame(y_pred, columns=['is_churned'])
    y_pred_df.to_csv('predictions.csv', index=None)
