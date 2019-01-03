import pickle

def load_cluster_data(if_union=False):
    # if_union == False means we only take the same annotation data as our dataset else we take the union of both annotations
    if if_union == False:
        file = open('../../data/pickle/clusters_separation.pickle', 'rb')
    else:
        file = open('../../data/pickle/clusters_separation_union.pickle', 'rb')
    training_clusters, validation_clusters, test_clusters = pickle.load(file)
    return training_clusters,validation_clusters,test_clusters

def build_distribution_dict(training_clusters,validation_clusters,test_clusters):

    distribution_dict = dict()
    for cluster in training_clusters:
        type_label = cluster[0]
        if type_label not in distribution_dict:
            distribution_dict[type_label] = 1
        else:
            distribution_dict[type_label] += 1
    for cluster in validation_clusters:
        type_label = cluster[0]
        if type_label not in distribution_dict:
            distribution_dict[type_label] = 1
        else:
            distribution_dict[type_label] += 1
    for cluster in test_clusters:
        type_label = cluster[0]
        if type_label not in distribution_dict:
            distribution_dict[type_label] = 1
        else:
            distribution_dict[type_label] += 1
    return distribution_dict


if __name__ == "__main__":
    training_clusters, validation_clusters, test_clusters = load_cluster_data()
    print(build_distribution_dict(training_clusters,validation_clusters,test_clusters))