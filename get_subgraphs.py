import pandas as pd
import osmnx as ox
from tqdm import tqdm
import torch
import networkx as nx
from torch_geometric.data import Data, Batch
import pickle
from multiprocessing import Pool, cpu_count



def process_node(node_id, graph):  
                
    # Create subgraph
    subgraph = nx.ego_graph(graph, node_id, radius=2)
    
    # Create node features
    num_nodes = len(subgraph.nodes())
    x = torch.zeros((num_nodes, 4), dtype=torch.float32)
    
    # Create node mapping for the subgraph
    node_map = {node: idx for idx, node in enumerate(subgraph.nodes())}
    
    # Add node features
    for i, (id, node_data) in enumerate(subgraph.nodes(data=True)):
        if 'pos' in node_data:
            x[i, 0:2] = torch.tensor(node_data['pos'], dtype=torch.float32)
        if id == node_id:
            x[i, 2] = 1.0  # Center node indicator
        if 'highway' in node_data:
            x[i, 3] = 1.0
    
    # Process edges
    edge_indices = []
    edge_attrs = []
    
    for u, v, data in subgraph.edges(data=True):
        if u in node_map and v in node_map:
            src_idx = node_map[u]
            dst_idx = node_map[v]
            
            # Add bidirectional edges
            edge_indices.append([src_idx, dst_idx])
            edge_indices.append([dst_idx, src_idx])
            
            # Create edge attributes
            edge_attr = torch.zeros(5, dtype=torch.float32)
            
            if 'length' in data:
                edge_attr[0] = data['length']
            if 'grade' in data:
                edge_attr[1] = data['grade']
            if 'highway' in data:
                edge_attr[2] = 1.0
            if 'oneway' in data and data['oneway']:
                edge_attr[3] = 1.0
            if 'weight' in data:
                edge_attr[4] = data['weight']
            
            edge_attrs.append(edge_attr)
            edge_attrs.append(edge_attr.clone())
    
    # Convert to tensors
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_attr = torch.stack(edge_attrs)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 5), dtype=torch.float32)
    
    # Create data object
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return graph_data



def main():
    df = pd.read_csv('data/processed_for_graph.csv')
    df = df[~df['nearest_nodes'].isna()]
    df['nearest_nodes'] = df['nearest_nodes'].apply(eval)

    node_to_i = {}
    for i, row in tqdm(df.iterrows()):
        new_nodes = [node for node in row['nearest_nodes'] if node not in node_to_i.keys()]
        for node in new_nodes:
            node_to_i[node] = row['i']


    print("loading graphs")
    i_to_graph = {}
    for i in tqdm(set(node_to_i.values())):
        i_to_graph[i] = pickle.load(open(f'test_cache/graph_{i}.pkl', 'rb'))


    print("preapring multiprocessing")
    # Prepare arguments for parallel processing
    args_list = []
    for node_id, i in tqdm(node_to_i.items()):
        args_list.append((node_id, i_to_graph[i]))
    
    # Use multiprocessing pool to process nodes in parallel
    num_cores = cpu_count() - 1  # Leave one core free
    with Pool(processes=num_cores) as pool:
        # Map function with progress bar
        results = list(tqdm(pool.starmap(process_node_wrapper, args_list), 
                            total=len(args_list), 
                            desc="Processing nodes"))
    
    print("Creating sub graph dict")
    # Create dictionary from results
    sub_graph_dict = {node_id: graph_data for node_id, graph_data in zip([arg[0] for arg in args_list], results)}
    
    print('Saving as pickle')
    # Save the results
    with open('data/sub_graph_dict.pkl', 'wb') as f:
        pickle.dump(sub_graph_dict, f)
    
    return sub_graph_dict

def process_node_wrapper(node_id, graph):
    """Wrapper function for process_node to handle exceptions"""
    try:
        return process_node(node_id, graph)
    except Exception as e:
        print(f"Error processing node {node_id}: {e}")
        # Return a simple empty graph data object
        return Data(x=torch.zeros((1, 4)), 
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, 5), dtype=torch.float32))

if __name__ == "__main__":
    main()






