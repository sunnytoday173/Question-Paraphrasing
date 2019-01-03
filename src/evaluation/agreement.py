# encoding:utf-8
import pandas as pd
import io
import sys

dframe1 = pd.read_excel('../../data/annotation/Annotator1.xlsx',header=0)
dframe2 = pd.read_excel('../../data/annotation/Annotator2.xlsx',header=0)

total = 0
classify_total = 0
classify_agreement = 0
agreement = 0
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

    a = [a1,a2,a3,a4,a5]
    b = [b1,b2,b3,b4,b5]

    classify_total += 1 # each cluster has a single label representing the question target
    if a0 == b0:
        classify_agreement += 1
    for i in range(len(a)):
        if type(a[i]) == str and type(b[i]) == str: # avoid empty situation
            total += 1

            annotation1 = a[i]
            annotation2 = b[i]

            if annotation1 == annotation2:
                agreement += 1
            '''
            else:
                print(annotation1,'|||',annotation2) # analyzing badcases
            '''

print('Classifier agreement count',classify_agreement)
print('Classifier agreement total', classify_total)
print('Classifier agreement', classify_agreement*1.0/classify_total)

print('Agreement count', agreement)
print('Agreement total', total)
print('Agreement', agreement * 1.0 / total)

# Classifier agreement 0.9354963306295867
# Agreement 0.5578914824547844