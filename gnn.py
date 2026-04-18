from torch.nn import Module, ReLU, Dropout, Sequential, ModuleList
from torch_geometric.nn import GATv2Conv, global_mean_pool, Linear
from torch.nn.functional import relu
from torch import cat
from config import DROPOUT


class GNN(Module):
    def __init__(self, num_features, num_classes, parameters):
        super(GNN, self).__init__()
        self.input_size = num_features
        self.output_size = num_classes
        self.dropout = DROPOUT

        self.layers_size = parameters['layers_size']
        self.num_hidden = parameters['hidden_layers']
        self.heads = parameters['heads']

        self.convs = ModuleList()
        self.convs.append(GATv2Conv(in_channels=self.input_size, out_channels=self.layers_size, heads=self.heads))
        # hidden layers
        for _ in range(self.num_hidden):
            self.convs.append(GATv2Conv(in_channels=self.layers_size * self.heads, out_channels=self.layers_size,
                                        heads=self.heads))

        self.convs.append(GATv2Conv(in_channels=self.layers_size * self.heads, out_channels=self.layers_size, heads=self.heads,
                                    concat=False))

        self.fully_connected = Sequential(
            Linear(in_channels=self.layers_size, out_channels=self.layers_size),
            ReLU(),
            Dropout(p=self.dropout),
            Linear(in_channels=self.layers_size, out_channels=self.output_size),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = relu(conv(x, edge_index))

        x = global_mean_pool(x, batch)
        x = self.fully_connected(x)

        return x
