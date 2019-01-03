# encoding:utf-8
import pandas as pd
import io
import sys
import re
import pickle
import random

random.seed(2018)

def dataset_separation(if_union=False):
    # if_union == False means we only take the same annotation data as our dataset else we take the union of both annotations
    dframe1 = pd.read_excel('../../data/annotation/Annotator1.xlsx', header=0)
    dframe2 = pd.read_excel('../../data/annotation/Annotator2.xlsx', header=0)

    clusters = []
    for i in range(2589):
        a0 = dframe1.iloc[i]['类别']
        a1 = dframe1.iloc[i]['句式1']
        a2 = dframe1.iloc[i]['句式2']
        a3 = dframe1.iloc[i]['句式3']
        a4 = dframe1.iloc[i]['句式4']
        a5 = dframe1.iloc[i]['句式5']
        b0 = dframe2.iloc[i]['类别']
        b1 = dframe2.iloc[i]['句式1']
        b2 = dframe2.iloc[i]['句式2']
        b3 = dframe2.iloc[i]['句式3']
        b4 = dframe2.iloc[i]['句式4']
        b5 = dframe2.iloc[i]['句式5']
        a = [a0, a1, a2, a3, a4, a5]
        b = [b0, b1, b2, b3, b4, b5]

        for j in range(5):
            if type(a[j]) == str:
                a[j] = re.sub('\ufeff', '', a[j])  # deal with specific character
            if type(b[j]) == str:
                b[j] = re.sub('\ufeff', '', b[j])  # deal with specific character

        #  For retrieval evaluation, we require that each cluster consisting of 5 sentences.
        if if_union == False:
            flag = True
            for j in range(6):
                if type(a[j]) != str or a[j] != b[j]:
                    flag = False
            if flag == True:
                clusters.append(tuple(a))
        else:
            flag = True
            for j in range(6):
                if type(a[j]) != str:
                    flag = False
            if flag == True:
                clusters.append(tuple(a))
            flag = True
            for j in range(6):
                if type(b[j]) != str:
                    flag = False
            if flag == True and tuple(b)!=tuple(a):
                clusters.append(tuple(b))

    random.shuffle(clusters)

    # The ratio of training:validation:test is 7:2:1
    training_clusters = clusters[:int(len(clusters) * 0.7)]
    validation_clusters = clusters[int(len(clusters) * 0.7):int(len(clusters) * 0.9)]
    test_clusters = clusters[int(len(clusters) * 0.9):]
    print("Training Size %d Validation Size %d Test Size %d" % (len(training_clusters),len(validation_clusters),len(test_clusters)))
    # Intersection separation result: Training Size 794 Validation Size 227 Test Size 114

    if if_union == False:
        file = open('../../data/pickle/clusters_separation.pickle', 'wb')
    else:
        file = open('../../data/pickle/clusters_separation_union.pickle', 'wb')

    pickle.dump((training_clusters, validation_clusters, test_clusters), file)

dataset_separation()



