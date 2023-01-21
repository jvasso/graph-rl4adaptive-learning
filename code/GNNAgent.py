import GLOBAL_VAR
from customObservation import CustomObservation

import torch
from torch import nn
from torch import cat, from_numpy, sum, mean
import torch.nn.functional as F

from torch_geometric.nn.dense import Linear, HeteroLinear
from torch_geometric.nn.conv import MessagePassing, GATConv, GCNConv, GeneralConv

import numpy as np

from typing import List


class GNNAgent(nn.Module):

    DENSE_LAYERS   = {"Linear":Linear, "HeteroLinear":HeteroLinear}
    CONV_LAYERS    = {"MessagePassing":MessagePassing, "GATConv":GATConv, "GCNConv":GCNConv, "GeneralConv":GeneralConv}
    MERGING_LAYERS = {"Mul":torch.mul, "Add":torch.add}
    STR2CLS = {**DENSE_LAYERS, **CONV_LAYERS, **MERGING_LAYERS}
    
    def __init__(self, hyperparams, kw_size, feedback_size, time_size):
        
        super(GNNAgent, self).__init__()
        self.nb_calls = 0

        general_params = hyperparams["general"]
        self.layer_params   = hyperparams["layers"]

        self.gnn_arch      = general_params["gnn_arch"]["arch"]
        self.feedback_arch = general_params["feedback_arch"]["arch"]
        self.time_arch     = "none" if general_params["time_arch"]=="none" else general_params["time_arch"]["arch"]
        self.middle_dim    = self.kw_embedding_size if (general_params["middle_dim"] == "standard") else general_params["middle_dim"]
        self.check_architectures()
        
        self.kw_embedding_size = kw_size
        self.feebdack_size     = feedback_size
        self.time_size         = time_size
        
        self.gnn_moduleList = self.build_layers(arch=self.gnn_arch, input_dim=self.kw_embedding_size)
        if self.feedback_arch is not None: self.feedback_moduleList = self.build_layers(arch=self.feedback_arch, input_dim=self.feebdack_size)
        if self.time_arch != "none": self.time_moduleList = self.build_layers(arch=self.time_arch, input_dim=self.time_size)

        in_channels  = self.middle_dim
        out_channels = 1
        self.last    = Linear(in_channels, out_channels, bias=True).double()
        
    
    def check_architectures(self):
        pass

    def build_layers(self, arch, input_dim):
        moduleList = torch.nn.ModuleList()
        for idx in range(len(arch)):
            layer_cls_name = arch[idx]["layer_cls"]
            layer_cls = GNNAgent.STR2CLS[layer_cls_name]
            if layer_cls_name == "MessagePassing":
                layer = layer_cls(**self.layer_params[layer_cls_name]).double()
            elif layer_cls_name=="Mul" or layer_cls_name=="Add":
                layer = None
            else :
                in_channels   = input_dim if idx==0 else self.middle_dim
                out_channels  = self.middle_dim
                channels_dict = {"in_channels":in_channels, "out_channels":out_channels}
                layer = layer_cls(**{**channels_dict.copy(), **self.layer_params[layer_cls_name]}).double()
            moduleList.append(layer)
        return moduleList

    
    @staticmethod
    def batch_processing(observations:List[CustomObservation]):
        X_nodes    = torch.cat([obs.get_node_features()           for obs in observations])
        X_feedback = torch.cat([obs.get_feedback_features()       for obs in observations])
        X_time     = torch.cat([obs.get_remaining_time_repeated() for obs in observations])
        
        nb_nodes_lengths = [0]+[obs.nb_nodes for obs in observations]
        cum_sum_nodes    = list(np.cumsum(nb_nodes_lengths))
        
        kw_ranges  = [[cum_sum_nodes[i-1]                        , cum_sum_nodes[i-1]+observations[i-1].nb_kw] for i in range(1,len(cum_sum_nodes))]
        doc_ranges = [[cum_sum_nodes[i-1]+observations[i-1].nb_kw, cum_sum_nodes[i-1]+observations[i-1].nb_nodes] for i in range(1,len(cum_sum_nodes))]
        nb_edges   = [ obs.nb_edge_indices for obs in observations]

        X_nodes_indices_slices = {}
        X_nodes_indices_slices["kw"]  = [slice(kw_range[0] , kw_range[1] ) for kw_range  in kw_ranges]
        X_nodes_indices_slices["doc"] = [slice(doc_range[0], doc_range[1]) for doc_range in doc_ranges]

        X_nodes_indices_tensors = {}
        X_nodes_indices_tensors["kw"]  = torch.cat([torch.tensor(range(kw_range[0] , kw_range[1] )) for kw_range  in kw_ranges ])
        X_nodes_indices_tensors["doc"] = torch.cat([torch.tensor(range(doc_range[0], doc_range[1])) for doc_range in doc_ranges])
        
        edge_indices = {}
        kw_extras  = torch.cat([ torch.cat([kw_ranges[i][0]*torch.ones((1,nb_edges[i])), doc_ranges[i][0]*torch.ones((1,nb_edges[i]))],dim=0) for i in range(len(observations))], dim=1)
        doc_extras = torch.cat([ torch.cat([doc_ranges[i][0]*torch.ones((1,nb_edges[i])), kw_ranges[i][0]*torch.ones((1,nb_edges[i]))],dim=0) for i in range(len(observations))], dim=1)
        edge_indices["kw2doc"] = (torch.cat([obs.get_edge_indices_kw2doc() for obs in observations], dim=1) + kw_extras ).long()
        edge_indices["doc2kw"] = (torch.cat([obs.get_edge_indices_doc2kw() for obs in observations], dim=1) + doc_extras).long()
        
        return X_nodes, X_feedback, X_time, X_nodes_indices_tensors, edge_indices, X_nodes_indices_slices
    

    @staticmethod
    def send_to_device(X_nodes: torch.tensor, X_feedback: torch.tensor, X_time: torch.tensor, X_nodes_indices_tensors: torch.tensor, edge_indices: torch.tensor):
        X_nodes    = X_nodes.to(GLOBAL_VAR.DEVICE)
        X_feedback = X_feedback.to(GLOBAL_VAR.DEVICE)
        X_time     = X_time.to(GLOBAL_VAR.DEVICE)
        assert set(X_nodes_indices_tensors.keys())=={"kw","doc"}
        X_nodes_indices_tensors["kw"]  = X_nodes_indices_tensors["kw"].to(GLOBAL_VAR.DEVICE)
        X_nodes_indices_tensors["doc"] = X_nodes_indices_tensors["doc"].to(GLOBAL_VAR.DEVICE)
        assert set(edge_indices.keys())=={"kw2doc", "doc2kw"}
        edge_indices["kw2doc"] = edge_indices["kw2doc"].to(GLOBAL_VAR.DEVICE)
        edge_indices["doc2kw"] = edge_indices["doc2kw"].to(GLOBAL_VAR.DEVICE)
        return X_nodes, X_feedback, X_time, X_nodes_indices_tensors, edge_indices
    
    
    def forward(self, X_nodes, X_feedback, X_time, X_nodes_indices_tensors, edge_indices):
        
        self.nb_calls += 1
        X_nodes, X_feedback, X_time, X_nodes_indices_tensors, edge_indices = GNNAgent.send_to_device(X_nodes, X_feedback, X_time, X_nodes_indices_tensors, edge_indices)
        
        x_feedback = torch.clone(X_feedback.double()).to(GLOBAL_VAR.DEVICE)
        x_time     = torch.clone(X_time.double()).to(GLOBAL_VAR.DEVICE)
        x_init     = torch.clone(X_nodes.double()).to(GLOBAL_VAR.DEVICE)
        x          = torch.zeros((X_nodes.shape[0], self.middle_dim)).double().to(GLOBAL_VAR.DEVICE)
        
        # feedback network
        if self.feedback_arch is not None:
            for idx in range(len(self.feedback_arch)):
                layer = self.feedback_moduleList[idx]
                x_feedback = layer(x_feedback) if idx==len(self.feedback_arch)-1 else F.relu(layer(x_feedback))
        else:
            raise Exception("feedback_arch is None.")
        
        # time network
        if self.time_arch != "none":
            for idx in range(len(self.time_arch)):
                layer = self.time_moduleList[idx]
                x_time = layer(x_time) if idx==len(self.time_arch)-1 else F.relu(layer(x_time))
        
        # gnn
        for idx in range(len(self.gnn_arch)):
            layer_dict     = self.gnn_arch[idx]
            layer_type     = layer_dict["layer_type"]
            layer_cls_name = layer_dict["layer_cls"]
            layer          = self.gnn_moduleList[idx]
            
            if idx == 0:
                assert self.is_dense_layer(layer_cls_name)
                nodes_indices = self.get_nodes_indices(layer_type, X_nodes_indices_tensors)
                x[nodes_indices] = F.relu(layer(x_init[nodes_indices]))
            else:
                if self.is_conv_layer(layer_cls_name):
                    if layer_cls_name == "MessagePassing":
                        edge_index = edge_indices[layer_type]
                        x = x + layer.propagate(edge_index, x=x)
                    else:
                        edge_index    = edge_indices[layer_type]
                        nodes_indices = self.get_nodes_indices(layer_type, X_nodes_indices_tensors)
                        x2 = F.relu(layer(x, edge_index=edge_index))
                        x[nodes_indices] = x2[nodes_indices]
                elif self.is_dense_layer(layer_cls_name):
                    nodes_indices = self.get_nodes_indices(layer_type, X_nodes_indices_tensors)
                    x[nodes_indices] = F.relu(layer(x[nodes_indices]))
                elif self.is_merging_layer(layer_cls_name):
                    nodes_indices = self.get_nodes_indices(layer_type, X_nodes_indices_tensors)
                    merging_operation = GNNAgent.MERGING_LAYERS[layer_cls_name]
                    if layer_type == "merge_f":
                        x3 = merging_operation(x[nodes_indices], x_feedback)
                        x[nodes_indices] = x3
                    elif layer_type == "merge_t":
                        x4 = merging_operation(x[nodes_indices], x_time)
                        x[nodes_indices] = x4
                    else:
                        raise Exception("Layer type '"+str(layer_type)+"' not supported.")
                else:
                    raise Exception("Layer class "+str(layer_cls_name)+" not supported.")

        return x
    

    def get_nodes_indices(self, layer_type, X_nodes_indices_tensors):
        if   layer_type in {"kw2kw", "doc2kw"}:
            nodes_indices = X_nodes_indices_tensors["kw"]
            return nodes_indices
        elif layer_type in {"doc2doc", "kw2doc", "merge_f", "merge_t"}:
            nodes_indices = X_nodes_indices_tensors["doc"]
            return nodes_indices
        else :
            Exception("Layer type "+str(layer_type)+" not supported.")
    

    def is_conv_layer(self, layer_cls_name):
        return layer_cls_name in GNNAgent.CONV_LAYERS.keys()
    
    def is_dense_layer(self, layer_cls_name):
        return layer_cls_name in GNNAgent.DENSE_LAYERS.keys()
    
    def is_merging_layer(self, layer_cls_name):
        return layer_cls_name in GNNAgent.MERGING_LAYERS.keys()
    
    def get_middle_dim(self):
        return self.middle_dim