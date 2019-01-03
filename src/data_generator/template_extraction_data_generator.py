import sys
import pickle
import random

from pytorch_pretrained_bert import BertTokenizer # Use BertTokenizer for tokenize

random.seed(2018)

tokenizer = BertTokenizer.from_pretrained('../../data/bert-base-chinese-vocab.txt') # Initialize tokenizer, maybe we can use a local way here

def load_cluster_data(if_union=False):
    # if_union == False means we only take the same annotation data as our dataset else we take the union of both annotations
    if if_union == False:
        file = open('../../data/pickle/clusters_separation.pickle', 'rb')
    else:
        file = open('../../data/pickle/clusters_separation_union.pickle', 'rb')
    training_clusters, validation_clusters, test_clusters = pickle.load(file)
    return training_clusters,validation_clusters,test_clusters

def data_generator(training_clusters,validation_clusters,test_clusters,if_union=False):

    cluster2pair(training_clusters,'training',if_union)
    cluster2pair(validation_clusters,'validation',if_union)
    cluster2pair(test_clusters,'test',if_union)

def cluster2pair(clusters,mode,if_union):

    filepath = "../../data/template_extraction_"
    if if_union:
        filepath += 'union_'
    filepath += mode
    filepath += '.txt'
    file = open(filepath,'w',encoding='utf-8')

    for cluster in clusters:
        # Mainly charcter based, but we have english character and numbers in our data
        # Idea: use tokenizer
        for i in range(1,len(cluster)):
            sentence = cluster[i]
            words = tokenizer.tokenize(sentence)

            components = []
            tags = []
            tag = 'O'
            for word in words:
                if word == '[' and tag == 'O':
                    tag = 'B'
                elif word == ']':
                    tag = 'O'
                elif tag == 'B':
                    components.append(word)
                    tags.append(tag)
                    tag = 'I'
                else:
                    components.append(word)
                    tags.append(tag)
            print(components, tags)
            str = ' '.join(components) + '|' + ' '.join(tags) + '\n'
            file.write(str)

if __name__ == "__main__":
    training_clusters, validation_clusters, test_clusters = load_cluster_data()
    data_generator(training_clusters,validation_clusters,test_clusters)
