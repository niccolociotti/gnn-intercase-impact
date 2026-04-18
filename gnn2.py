from torch.nn import Module, ReLU, Dropout, Sequential, Linear, ModuleList
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch import cat
from config import DROPOUT

class GNN(Module):
    def __init__(self, num_features, num_classes, parameters):
        super(GNN, self).__init__()
        
        self.dropout = DROPOUT
        self.layers_size = parameters['layers_size']
        self.heads = parameters['heads']
        self.num_hidden = parameters['hidden_layers']

        #--- RAMO 1: Traccia (GNN1 - Sequenziale) ---
        # Input: num_features (dalla traccia)
        self.conv_trace = ModuleList()       
        self.conv_trace.append(GATv2Conv(num_features, self.layers_size, heads=self.heads))

        for _ in range(self.num_hidden):
            self.conv_trace.append(
                GATv2Conv(self.layers_size * self.heads, self.layers_size, heads=self.heads))
            
        self.conv_trace.append(GATv2Conv(in_channels=self.layers_size * self.heads, out_channels=self.layers_size, heads=self.heads,
                                    concat=False))
        
        # --- RAMO 2: Contesto (GNN2 - Concorrente) ---
        # Input: num_features (dal contesto - assumiamo stesso dimensionamento OHE)
        self.conv_ctx = ModuleList()
        self.conv_ctx.append(GATv2Conv(num_features, self.layers_size, heads=self.heads))

        for _ in range(self.num_hidden):
            self.conv_ctx.append(
                GATv2Conv(self.layers_size * self.heads, self.layers_size, heads=self.heads)) 
            
        self.conv_ctx.append(GATv2Conv(in_channels=self.layers_size * self.heads, out_channels=self.layers_size, heads=self.heads,
                                    concat=False))
        
        # --- FUSIONE ---
        # Concateniamo i due vettori: dim_traccia + dim_contesto
        fusion_size = self.layers_size * 2 
        
        self.fully_connected = Sequential(
            Linear(fusion_size, self.layers_size),
            ReLU(),
            Dropout(p=self.dropout),
            Linear(self.layers_size, num_classes)
        )

    def forward(self, data):
        # --- RAMO 1: Traccia ---
        x, edge_index, batch = data.x, data.edge_index, data.batch
    
        for conv in self.conv_trace:
            x = conv(x, edge_index).relu()
    
        out_trace = global_mean_pool(x, batch)
    
        # --- RAMO 2: Contesto ---
        x_ctx, edge_index_ctx, batch_ctx = (
            data.x_ctx,
            data.edge_index_ctx,
            data.x_ctx_batch
        )
    
        h = x_ctx
        for conv in self.conv_ctx:
            h = conv(h, edge_index_ctx).relu()
    
        out_ctx = global_mean_pool(h, batch_ctx)
    
        # --- FUSIONE ---
        combined = cat([out_trace, out_ctx], dim=1)
    
        # --- CLASSIFICAZIONE ---
        out = self.fully_connected(combined)
    
        return out
