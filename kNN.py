import numpy as np
from math import sqrt
from collections import Counter

def kNN_classify(k,X_train,y_train,x):

    distance = [sqrt(np.sum((x-x_train)**2)) for x_train in X_train]
    nearest = np.argsort(distance)

    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)

    return votes.most_common(1)[0][0]
