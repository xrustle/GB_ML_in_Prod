def xgb_predict(clf, x):
    pred_proba = clf.predict_proba(x)
    pred = clf.predict(x)

    return pred_proba, pred
