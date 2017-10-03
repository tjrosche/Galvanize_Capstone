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
from sklearn.metrics import log_loss, roc_auc_score

def logistic_AUC_determine_num_factors(X_train, y_train, X_test, y_test):
    logit = LogisticRegressionCV(Cs=[0.5,0.8,1,2,3,5], cv=5, n_jobs=-1)
    logit_fit= logit.fit(X_train, y_train)
    pred_prob = logit_fit.predict_proba(X_test)
    C= logit_fit.C_
    return C, roc_auc_score(y_score=pred_prob[:,1], y_true=y_test)

def TF_vectorizer(traindf, testdf, max_features=20000):
    count_vect = CountVectorizer(min_df = 1, ngram_range=(1,2), max_features=max_features)
    X_train_counts = count_vect.fit_transform(traindf["Text_Clean"])
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_new_counts = count_vect.transform(testdf["Text_Clean"])
    X_test_tfidf = tfidf_transformer.transform(X_new_counts)
    y_train = traindf["Useful"]
    y_test = testdf["Useful"]
    return count_vect, X_train_tfidf, y_train, X_test_tfidf, y_test

def cost_finder(train_df, test_df):
    feature_list = [500, 1000, 5000, 10000, 15000, 20000, 25000, 30000,40000]
    cost_list=[]
    C_list=[]
    for i in feature_list:
        count_vect, X_train_tfidf, y_train, X_test_tfidf, y_test = TF_vectorizer(train_df, test_df, max_features=i)
        C, cost=logistic_AUC_determine_num_factors(X_train_tfidf, y_train, X_test_tfidf, y_test)
        cost_list.append(cost)
        C_list.append(C)
    return feature_list, cost_list, C_list

if __name__ == "__main__":
    df_keepers = pd.read_pickle('food_keepers_cos_sim_train_balanced_split6.pkl')
    different_test_sizes = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
    total_cost_lists = []
    total_C_lists =[]
    feature_list = [500, 1000, 5000, 10000, 15000, 20000, 25000, 30000,40000]
    for test_size in different_test_sizes:
        train, test = train_test_split(df_keepers, test_size=test_size)
        feature_list, cost_list, C_list= cost_finder(train,test)
        total_cost_lists.append(cost_list)
        total_C_lists.append(C_list)
    df_costs = pd.DataFrame.from_records(total_cost_lists, columns=feature_list)
    df_Cs = pd.DataFrame.from_records(total_C_lists, columns=feature_list)
    df_costs.to_pickle('AUC_costs_df.pkl')
    df_Cs.to_pickle('AUC_Cs_df.pkl')
