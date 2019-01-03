import sys
import pickle
import random

from pytorch_pretrained_bert import BertTokenizer # Use BertTokenizer for tokenize

random.seed(2018)

tokenizer = BertTokenizer.from_pretrained('../../data/bert-base-chinese-vocab.txt') # Initialize tokenizer

def load_cluster_data(if_union=False):
    # if_union == False means we only take the same annotation data as our dataset else we take the union of both annotations
    if if_union == False:
        file = open('../../data/pickle/clusters_separation.pickle', 'rb')
    else:
        file = open('../../data/pickle/clusters_separation_union.pickle', 'rb')
    training_clusters, validation_clusters, test_clusters = pickle.load(file)
    return training_clusters,validation_clusters,test_clusters

def template_generation(sentence):
    # generate template from sentence
    words = tokenizer.tokenize(sentence)
    components = []
    tag = 'O'
    for word in words:
        if word == '[' and tag == 'O':
            tag = 'B'
            components.append('#')  # use which character as placeholder may worth considering
        elif word == ']':
            tag = 'O'
        elif tag == 'B':
            tag = 'I'
        elif tag == 'O':
            components.append(word)
    return components

def build_template_pickle(clusters,if_union=False):
    filepath = "../../data/pickle/template_set"
    if if_union:
        filepath += '_union'
    filepath += '.pickle'
    file = open(filepath, 'wb')
    template_set = set()
    for cluster in clusters:
        for i in range(1, len(cluster)):
            sentence = cluster[i]
            components = template_generation(sentence)
            template_set.add(' '.join(components))
    pickle.dump(template_set,file)

if __name__ == "__main__":
    training_clusters, validation_clusters, test_clusters = load_cluster_data()
    build_template_pickle(training_clusters + validation_clusters + test_clusters)
