import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/Helpdesk_no_resources_processed.g")

print(df.head())

case_id = df['case_id'].iloc[1]

nodes = df[(df['case_id'] == case_id) & (df['type'] == 'v')]
edges = df[(df['case_id'] == case_id) & (df['type'] == 'e')]

print("Nodi:")
print(nodes)
print("\nArchi:")
print(edges)

G = nx.DiGraph()

for _, r in nodes.iterrows():
    G.add_node(r['node1'], label=r['activity'])

for _, r in edges.iterrows():
    G.add_edge(r['node1'], r['node2'])

pos = nx.spring_layout(G)
labels = nx.get_node_attributes(G, 'label')

nx.draw(G, pos, node_size=2000)
nx.draw_networkx_labels(G, pos, labels)

print("Nodi:", G.number_of_nodes())
print("Archi:", G.number_of_edges())
print(list(G.nodes())[:10])
print(list(G.edges())[:10])


plt.show()
