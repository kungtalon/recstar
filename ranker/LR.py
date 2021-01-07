import numpy as np
import pandas as pd
from data.loader import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

penalty_param_range = (0.01, 0.03, 0.1, 0.3, 1, 3, 10)
params = []
# params += [['l1', 'liblinear', c] for c in penalty_param_range]
# params += [['l2', 'liblinear', c] for c in penalty_param_range]
params = [['l1', 'liblinear', 0.3]]

def cal_auc(y, pred):
    fpr, tpr, _ = roc_curve(y, pred, pos_label=1)
    return auc(fpr, tpr)

def main():
    dataLoader = DataLoader()
    dataLoader.use_small_dataset()
    
    X_train, y_train = dataLoader.train_set
    X, X_, y, y_ = train_test_split(X_train, y_train, test_size=0.15)

    best_auc = 0
    for param_set in params:
        print(param_set)
        clf = LogisticRegression(penalty=param_set[0],
                                 solver=param_set[1],
                                 C=param_set[2]).fit(X, y)
        pred = clf.predict_proba(X_)[:, 1]
        auc = cal_auc(y_, pred)
        if auc > best_auc:
            best_auc = auc
            best_model = clf
            print("auc : {}".format(best_auc))

    print(best_model)
    XX, yy = dataLoader.test_set
    pred = best_model.predict_proba(XX)[:,1]
    print(cal_auc(yy, pred))
    # submission = pd.read_csv('./data/sampleSubmission.gz', compression='gzip', index_col='id')
    # submission[submission.columns[0]] = p[:,1]
    # submission.to_csv('submission.csv')

if __name__ == '__main__':
    main()
    