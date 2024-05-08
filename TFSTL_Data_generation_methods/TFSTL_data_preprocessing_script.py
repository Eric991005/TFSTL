# Import necessary libraries
import pandas as pd
import numpy as np
import networkx as nx
import os
from ge.classify import read_node_label, Classifier
from ge import Struc2Vec

# Relative directory path
################# You should replace the path here ##################
directory = '/root/autodl-tmp/TFSTL_Upload_Maintenance/data/MYDATA/new_export'
if not os.path.exists(directory):
    os.makedirs(directory)
    print("Directory created:", directory)

# Reading the data
################# You should replace the csv here ##################
df = pd.read_csv(f'{directory}/export_new.csv', index_col=0)
df.index = pd.to_datetime(df.index)

# Convert data to NumPy array and save as npz
data = df.values
np.savez(f'{directory}/flow.npz', result=data)
print('NumPy array saved as npz successfully.')

# Calculate and save correlation matrix
corr_matrix = np.corrcoef(df, rowvar=False)
np.save(f'{directory}/corr_adj.npy', corr_matrix)
print('Correlation matrix saved successfully.')

# Generate and save edge list
edge_list = []
for i in range(corr_matrix.shape[0]):
    for j in range(i+1, corr_matrix.shape[1]):
        weight = corr_matrix[i, j]
        edge_list.append((i, j, weight))
with open(f'{directory}/data.edgelist', 'w') as f:
    for edge in edge_list:
        f.write('{} {} {}\n'.format(edge[0], edge[1], edge[2]))
print('Edge list saved successfully.')

# Read the graph, train the model, and save embeddings
G = nx.read_edgelist(f'{directory}/data.edgelist', create_using=nx.DiGraph(), nodetype=None, data=[('weight', float)])
model = Struc2Vec(G, 10, 80, workers=4, verbose=40)
model.train(embed_size=128)
embeddings = model.get_embeddings()

# Convert embeddings to numpy array and save
embedding_array = np.array(list(embeddings.values()))
np.save(f'{directory}/128_corr_struc2vec_adjgat.npy', embedding_array)
print('Embedding array saved successfully.')
