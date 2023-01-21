import PATHS

import utils_files

from typing import Any, Dict
import numpy as np
import torch
import copy


class CorpusGraph():
    
    def __init__(self, corpus_name:str, node_features:dict):
        """
        graph_dict: {doc_id: [kw_id]}
        """
        self.corpus_name = corpus_name
        graph_dict, kw2vec, kw2str, doc2kw_str = CorpusGraph.preprocess_graph_data(corpus_name, node_features)
        
        self.graph_dict = graph_dict
        self.kw2vec = kw2vec
        self.kw2str = kw2str
        self.doc2kw_str = doc2kw_str
        
        self.kw_str_list = [kw2str[kw_idx] for kw_idx in self.kw2str]
        self.kw_list  = list(self.kw2vec.keys())
        self.doc_list = list(self.graph_dict.keys())
        assert self.kw_list  == list(range(len(self.kw_list)))
        assert self.doc_list == list(range(len(self.doc_list)))

        self.nb_kw  = len(self.kw_list)
        self.nb_doc = len(self.doc_list)

        self.kw2doc = self.build_kw2doc()
        self.kw_features, self.doc_features = self.build_features_matrices(node_features=node_features)
        self.doc2kw_edge_idx, self.kw2doc_edge_idx = self.build_edge_indices()

        self.kw_size = self.kw_features.shape[1]
    

    @staticmethod
    def preprocess_graph_data(corpus_name, node_features):
        corpus_data_path = PATHS.CORPUS_DATA+"/"+corpus_name
        corpus_file = corpus_data_path+"/corpus.json"
        doc2kw_str = CorpusGraph.load_course_data(corpus_file)
        CorpusGraph.check_kw_dict(doc2kw_str)
        kw_str2vec = CorpusGraph.load_kw_embeddings(node_features=node_features, corpus=corpus_name)
        graph_dict, kw_str2idx, kw_idx2str = CorpusGraph.build_graph_dict(doc2kw_str)
        kw_idx2vec = CorpusGraph.build_kw2vec(kw_idx2str, kw_str2vec)
        return graph_dict, kw_idx2vec, kw_idx2str, doc2kw_str
    
    @staticmethod
    def load_kw_embeddings(node_features:dict, corpus:str):
        corpus_data_dir = PATHS.CORPUS_DATA
        feature_params = node_features["features_params"]
        feature_type   = feature_params["feature_type"]
        folder         = feature_params["folder"]
        caracteristics = {"corpus":corpus, "feature_type":feature_type, "folder":folder}
        corpus_data_structure = PATHS.CORPUS_DATA_STRUCTURE
        subfolder = corpus_data_dir+"/"+"/".join([caracteristics[key] for key in corpus_data_structure])
        if feature_type == "one_hot_encoding":
            one_hot_encoding_type = feature_params["one_hot_encoding_type"]
            filename = subfolder + "/" + one_hot_encoding_type + ".npy"
        elif feature_type == "embedding":
            filename = subfolder + "/embeddings.npy"
        else:
            raise Exception("Feature type "+str(feature_type)+" not recognized.")
        words_embeddings = CorpusGraph.load_embeddings(filename)
        return words_embeddings

    @staticmethod
    def load_embeddings(filename):
        data=np.load(filename, allow_pickle=True)
        words_embeddings = data[()]
        return words_embeddings
    
    @staticmethod
    def load_course_data(filename):
        kw_json = utils_files.load_json_file(filename)
        kw_dict = CorpusGraph.preprocess_kw_dict(kw_json)
        return kw_dict
    
    @staticmethod
    def preprocess_kw_dict(kw_dict):
        kw_dict_output = {}
        for resource_id in kw_dict:
            kw_dict_output[int(resource_id)] = copy.deepcopy(kw_dict[resource_id])
        return kw_dict_output
    
    @staticmethod
    def check_kw_dict(doc2kw_str):
        # Make sure that the resources are in order, starting from 0.
        doc_list = list(doc2kw_str.keys())
        assert doc_list == list(range(len(doc_list)))
    
    def build_graph_dict(doc2kw_str):
        graph_dict, kw_str2idx, kw_idx2str = {}, {}, {}
        kw_str_list = []
        current_idx = 0
        for doc in doc2kw_str:
            graph_dict[doc] = []
            for kw_str in doc2kw_str[doc]:
                if kw_str in kw_str_list:
                    kw_idx = kw_str2idx[kw_str]
                    graph_dict[doc].append(kw_idx)
                else:
                    kw_str_list.append(kw_str)
                    kw_str2idx[kw_str] = current_idx
                    kw_idx2str[current_idx] = kw_str
                    kw_idx = current_idx
                    graph_dict[doc].append(kw_idx)
                    current_idx += 1
        return graph_dict, kw_str2idx, kw_idx2str
    
    @staticmethod
    def build_kw2vec(kw_idx2str, kw_str2vec):
        kw_idx2vec = {}
        for kw_idx in list(kw_idx2str.keys()):
            kw_str = kw_idx2str[kw_idx]
            kw_idx2vec[kw_idx]=kw_str2vec[kw_str]
        return kw_idx2vec

    def build_features_matrices(self, node_features):
        normalize    = node_features["normalize"]
        doc_encoding = node_features["doc_encoding"]
        embeddings_size = self.kw2vec[self.kw_list[0]].shape[0]

        kw_features_matrix   = torch.cat([torch.unsqueeze(torch.tensor(np.float64(self.kw2vec[kw])),dim=0) for kw in self.kw2vec.keys()],dim=0)

        if normalize:
            means = kw_features_matrix.mean(0, keepdim=True)
            deviations = kw_features_matrix.std(0, keepdim=True)
            means_transformed      = torch.cat([torch.clone(means)      for j in range(kw_features_matrix.shape[0])], dim=0)
            deviations_transformed = torch.cat([torch.clone(deviations) for j in range(kw_features_matrix.shape[0])], dim=0)
            kw_features_matrix = torch.div(kw_features_matrix - means_transformed, deviations_transformed)
        
        if doc_encoding == "zero":
            docs_features_matrix = torch.zeros((self.nb_doc, embeddings_size))
        elif doc_encoding == "one-hot-encoding":
            eye_matrix           = torch.eye(n=self.nb_doc, m=embeddings_size)
            perm                 = torch.randperm(eye_matrix.size(0))
            docs_features_matrix = eye_matrix[perm]
        else:
            raise Exception("Error: doc encoding '"+str(doc_encoding)+"' not supported.")
        
        return kw_features_matrix, docs_features_matrix
    

    def build_edge_indices(self):
        nb_edges = self.count_edges()
        doc2kw_edge_idx = torch.zeros((2,nb_edges))
        kw2doc_edge_idx = torch.zeros((2,nb_edges))
        i = 0
        for doc in list(self.graph_dict.keys()):
            for kw in self.graph_dict[doc]:
                doc2kw_edge_idx[0,i] = doc
                doc2kw_edge_idx[1,i] = kw
                i+=1
        j = 0
        for kw in list(self.kw2doc.keys()):
            for doc in self.kw2doc[kw]:
                kw2doc_edge_idx[0,j] = kw
                kw2doc_edge_idx[1,j] = doc
                j+=1
        return doc2kw_edge_idx, kw2doc_edge_idx
    

    def build_kw2doc(self):
        kw2doc = {kw:[] for kw in self.kw_list}
        for doc in list(self.graph_dict.keys()):
            for kw in self.graph_dict[doc]:
                kw2doc[kw].append(doc)
        return kw2doc
    

    def count_edges(self):
        count = 0
        for doc in list(self.graph_dict.keys()):
            count += len(self.graph_dict[doc])
        return count

    def test(self):
        edge_index_kw2lo = [[],[]]
        edge_index_lo2kw = [[],[]]
        
        for kw_idx in self.kw_indices:
            for lo_idx in self.kw_idx_2_lo_idx[kw_idx]:
                edge_index_kw2lo[0].append(kw_idx)
                edge_index_kw2lo[1].append(lo_idx)
                
        for lo_idx in self.lo_indices:
            for kw_idx in self.lo_idx_2_kw_idx[lo_idx]:
                edge_index_lo2kw[0].append(lo_idx)
                edge_index_lo2kw[1].append(kw_idx)

    def get_nb_doc(self):
        return self.nb_doc
    
    def get_nb_kw(self):
        return self.nb_kw

    def get_kw_features(self):
        return torch.clone(self.kw_features)
    
    def get_kw_size(self):
        return self.kw_size
    
    def get_doc_features(self):
        return torch.clone(self.doc_features)
    
    def get_doc2kw_edge_idx(self):
        return torch.clone(self.doc2kw_edge_idx)
    
    def get_kw2doc_edge_idx(self):
        return torch.clone(self.kw2doc_edge_idx)

    def get_graph_dict(self):
        return copy.deepcopy(self.graph_dict)
    
    def get_kw_list(self):
        return copy.deepcopy(self.kw_list)
    
    def get_doc_list(self):
        return copy.deepcopy(self.doc_list)
    
    def get_doc2kw_str(self):
        return copy.deepcopy(self.doc2kw_str)
    
    def get_kw_str_list(self):
        return copy.deepcopy(self.kw_str_list)