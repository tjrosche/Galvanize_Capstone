import pandas as pd
import numpy as np
import nltk
import string
import re
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
import json
import gzip
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import log_loss


def logistic_AUC_determine_num_factors(X_train, y_train, X_test, y_test):
    logit = LogisticRegressionCV(Cs=[0.5,0.8,1,2,3,5], cv=5, n_jobs=-1)
    logit_fit= logit.fit(X_train, y_train)
    pred_prob = logit_fit.predict_proba(X_test)
    C= logit_fit.C_
    return logit_fit, roc_auc_score(y_score=pred_prob[:,1], y_true=y_test)


if __name__ == "__main__":
    X_train = np.load('X_trainT.npy')
    X_test = np.load('X_testT.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    total_AUC_list = []

    feature_list = [500, 1000, 1500, 2000, 2250, 2500, 2750, 3000, 3500, 4000]
    for pcs in feature_list:
        X_testT =X_test[:,:pcs]
        X_trainT = X_train[:,:pcs]
        log_fit, AUC=logistic_AUC_determine_num_factors(X_trainT, y_train, X_testT, y_test)
        total_AUC_list.append(AUC)

    sr_AUC = pd.Series.from_array(total_AUC_list)
    sr_AUC.to_pickle('AUC_sr.pkl')
    filename = 'logfit_4000'
    pickle.dump(log_fit, open(filename, 'wb'))
