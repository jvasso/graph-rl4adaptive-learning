import torch
import pprint

class CustomObservation:

    def __init__(self, 
                kw_features: torch.Tensor,
                doc_features: torch.Tensor,
                feedback_features: torch.Tensor,
                edge_indices_doc2kw: torch.Tensor,
                edge_indices_kw2doc: torch.Tensor,
                remaining_time: torch.Tensor):
        
        self.kw_features         = kw_features
        self.doc_features        = doc_features
        self.feedback_features   = feedback_features
        self.edge_indices_doc2kw = edge_indices_doc2kw
        self.edge_indices_kw2doc = edge_indices_kw2doc
        self.remaining_time      = remaining_time

        self.nb_feedback     = self.feedback_features.shape[0]
        self.nb_kw           = self.kw_features.shape[0]
        self.nb_doc          = self.doc_features.shape[0]
        self.nb_edge_indices = self.edge_indices_kw2doc.shape[1]
        
        self.nb_nodes      = self.nb_kw + self.nb_doc
        self.node_features = torch.cat([self.kw_features, self.doc_features])

    
    def get_node_features(self):
        return torch.clone(self.node_features)
    
    def get_feedback_features(self):
        return torch.clone(self.feedback_features)
    
    def get_remaining_time(self):
        return torch.clone(self.remaining_time)
    
    def get_remaining_time_repeated(self):
        return torch.clone(self.remaining_time.clone().repeat(self.nb_doc ,1))
    
    def get_edge_indices_kw2doc(self):
        return torch.clone(self.edge_indices_kw2doc)
    
    def get_edge_indices_doc2kw(self):
        return torch.clone(self.edge_indices_doc2kw)

    def get_nb_doc(self):
        return self.nb_doc
    
    def get_nb_kw(self):
        return self.nb_kw

    @staticmethod
    def print_list(observations_list):
        for obs in observations_list:
            pprint.pprint(obs.get_feedback_features())