import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os


def store_graph_features(data_array, input_directory, dataset_name):
    print ("extracting the 6 features for dataset in:",input_directory)
    MG = nx.MultiGraph()
    MG.add_weighted_edges_from(data_array, weight=1)
    MG_simple = nx.Graph(MG)


    dg = nx.degree(MG_simple) #for  multi  graph gives the same address because it does not count relations between two nodes in calculating degree
    dg_per_graph = dg    #/ node_count could not divide, todo:later do it at load time
    np.save(input_directory + dataset_name + "degree", dg_per_graph)

    cen = nx.eigenvector_centrality_numpy(MG)  #this one gives dictionary of velaue per node
    np.save(input_directory + dataset_name + "centrality",cen)

    pg_rank = nx.pagerank(MG_simple, alpha=0.85) # not implemented for muktigraphs
    np.save(input_directory + dataset_name + "pagerank", pg_rank)

    clos = nx.closeness_centrality(MG)
    np.save(input_directory + dataset_name + "closeness",clos)

    netw = nx.betweenness_centrality(MG_simple)
    np.save(input_directory + dataset_name + "betweenness",netw)

    katz = nx.katz_centrality_numpy(MG_simple)#max_iter=1000000000  # not implemented for multigraph type
    np.save(input_directory + dataset_name + "katz", katz)

# def generate_id_file(input_triple_file_path):
#     train_data_ = np.loadtxt(open(input_triple_file_path+"train.txt", "rb"), delimiter="\t", skiprows=0,
#                              dtype="str")
#
#     entity2id_ = np.zeros(train_data_.shape[0], dtype='int')
#     relation2id_ = np.zeros(relation2id_.shape[0], dtype='int')
#     entity_dic = {}
#     relation_dic = {}
#     for entity_line in entity2id_:
#         entity_dic[entity_line[1]] = int(entity_line[0])
#         self.entity2id_[int(entity_line[0])] = int(entity_line[0])
#     for rel_line in relation2id_:
#         relation_dic[rel_line[1]] = int(rel_line[0])
#         self.relation2id_[int(rel_line[0])] = int(rel_line[0])
#
#     self.train_data_ = np.zeros(train_data.shape, dtype='int')
#     self.test_data_ = np.zeros(test_data.shape, dtype='int')
#     self.validation_data_ = np.zeros(validation_data.shape)
#     for t_num in range(0, self.train_data_.shape[0]):
#         self.train_data_[t_num, 0] = entity_dic[train_data[t_num, 0]]
#         self.train_data_[t_num, 1] = entity_dic[train_data[t_num, 2]]
#         self.train_data_[t_num, 2] = relation_dic[train_data[t_num, 1]]

def make_features_for_pattern_datasets():
    input_directory = "../data/"
    filenames = os.listdir(input_directory)  # get all files' and folders' names in the current directory

    result = []
    for filename in filenames:  # loop through all the files and folders
        if os.path.isdir(os.path.join(input_directory, filename)):  # check whether the current object is a folder
            result.append(filename)

    result.sort()

    for dataset_name in result:
        #generate_id_file(input_directory + dataset_name+ "/train.txt")
        train_data_file = input_directory + dataset_name+  "/train.txt"
        out_dir_path =  input_directory + dataset_name+ "/train_node_features"
        if os.path.isdir(out_dir_path):
            print("Directory %s to create already exists." %out_dir_path)
        else:
            try:
                os.mkdir(out_dir_path)
            except OSError:
                print("Creation of the directory %s failed" % out_dir_path)
                exit()

        out_dir_path = out_dir_path + "/"
        train_data_ = np.loadtxt(open(train_data_file, "rb"), delimiter="\t", skiprows=0,
                                dtype="str")
        train_data_[:, [2, 1]] = train_data_[:, [1, 2]] #it's column must be in shape [entity, entity,relation]
        store_graph_features(train_data_, out_dir_path, "")

make_features_for_pattern_datasets()
