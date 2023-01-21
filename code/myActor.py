import GLOBAL_VAR
from GNNAgent import GNNAgent
from customObservation import CustomObservation

from typing import Any, Dict, Optional, Sequence, Tuple, Union
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.nn.dense import Linear


class MyActor(nn.Module):
    """Simple actor network.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param bool softmax_output: whether to apply a softmax layer over the last
        layer's output.
    """

    def __init__(
        self,
        gnnAgent: GNNAgent
    ) -> None:
        super().__init__()
        self.gnnAgent = gnnAgent
        
        in_channels  = gnnAgent.get_middle_dim()
        out_channels = 1
        self.last    = Linear(in_channels, out_channels, bias=True).double()
    

    def forward(self, observations:List[CustomObservation], state: Any = None, info: Dict[str, Any] = {}):        
        batch_size  = len(observations)
        nb_docs     = [obs.get_nb_doc() for obs in observations]
        nb_docs_max = max(nb_docs)
        X_nodes, X_feedback, X_time, X_nodes_indices_tensors, edge_indices, X_nodes_indices_slices = GNNAgent.batch_processing(observations)
        x = self.gnnAgent(X_nodes, X_feedback, X_time, X_nodes_indices_tensors, edge_indices)
        out = self.last(x)
        out_squeezed = torch.squeeze(out)
        docs_slices  = X_nodes_indices_slices["doc"]
        out_padded   = torch.cat( [ torch.cat( [ out_squeezed[docs_slices[i]], torch.full((nb_docs_max-nb_docs[i],),-float('inf')).to(GLOBAL_VAR.DEVICE) ] ).unsqueeze(dim=0) for i in range(batch_size) ], dim=0)
        logits  = F.softmax(out_padded, dim=1)
        return logits, state

