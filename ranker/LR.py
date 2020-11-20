import numpy as np
import pandas as pd
from data.loader import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

penalty_param_range = (0.01, 0.03, 0.1, 0.3, 1, 3, 10)
params = []
params += [['l1', 'liblinear', c] for c in penalty_param_range]
params += [['l2', 'liblinear', c] for c in penalty_param_range]

def main():
    dataLoader = DataLoader()
    dataLoader.use_small_dataset()
    
    X_train, y_train = dataLoader.train_set
    X, X_, y, y_ = train_test_split(X_train, y_train, test_size=0.15)

    best_loss = np.inf
    for param_set in params:
        print(param_set)
        clf = LogisticRegression(penalty=param_set[0],
                                 solver=param_set[1],
                                 C=param_set[2]).fit(X, y)
        pred = clf.predict_proba(X_)[:, 1]
        loss = log_loss(y_, pred)
        if loss < best_loss:
            best_loss = loss
            best_model = clf
            print("loss : {}".format(best_loss))

    print(best_model)
    XX = dataLoader.test_set
    p = best_model.predict_proba(XX)
    submission = pd.read_csv('./data/sampleSubmission.gz', compression='gzip', index_col='id')
    submission[submission.columns[0]] = p[:,1]
    submission.to_csv('submission.csv')

if __name__ == '__main__':
    main()
    