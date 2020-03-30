from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb


def balancing(X, y):
    X, y = SMOTE(random_state=42, sampling_strategy=0.3).fit_sample(X, y)
    return X, y


def scale(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def x_and_y(train):
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
