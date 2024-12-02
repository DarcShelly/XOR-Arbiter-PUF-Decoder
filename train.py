import numpy as np
import sklearn
from scipy.linalg import khatri_rao

def my_fit( X_train, y_train,  model='LR', penalty='l2', C=6.0, tol=0.0001):
    if model == 'LR':
        from sklearn.linear_model import LogisticRegression
        feat = my_map(X_train)
        clf = LogisticRegression(dual=False, C=C, penalty=penalty, tol=tol).fit(feat, y_train)
        w, b = clf.coef_.T.flatten(), clf.intercept_
        return w, b
    
    #else
    from sklearn.svm import LinearSVC
    feat = my_map(X_train)
    clf = LinearSVC(dual=False, penalty=penalty, C=C, tol=tol).fit(feat, y_train)
    w, b = clf.coef_.T.flatten(), clf.intercept_
    return w, b

def my_map( X ):
    X = 2*X-1
    n=len(X)
    n_=len(X[0])
    X = np.flip(np.cumprod(np.flip(X, axis=1), axis=1), axis=1)
    X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    n=len(X)
    m=len(X[0])
    feat = np.empty((n, int(m*(m-1)/2)), dtype = X.dtype)
    ind = 0
    for i in range(m):
        for j in range(i+1, m):
            feat[:, ind] = 2 * X[:, i] * X[:, j]
            ind+=1
    return feat
