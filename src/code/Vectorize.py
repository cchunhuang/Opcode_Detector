import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def Vectorize(sequence):
    seq = ''
    for i in sequence:
        seq += str(i + ' ')
    X = list()
    X.append(seq)
    vectorizer_count = CountVectorizer(ngram_range = (2, 4))
    x_count = vectorizer_count.fit_transform(X)
    feature_list = vectorizer_count.get_feature_names()
    top_feature= np.load("./top_features_1.npy")
    tmp = []
    for j in range(len(top_feature)):
        if(top_feature[j] in feature_list):
            tmp.append(x_count[:,feature_list.index(top_feature[j])][0,0])
        else:
            tmp.append(0)
    tmp = np.asarray(tmp)
    tmp = tmp.reshape(1,-1)
    return(tmp)