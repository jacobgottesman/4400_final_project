from tqdm import tqdm
import osmnx as ox
import pickle
import pandas as pd
from multiprocessing import Pool, cpu_count

# Define the process_route function
def process_route(row_data):
    lats, longs, idx = row_data['latitude'], row_data['longitude'], row_data['i']
    nearest_nodes = []

    try:
        graph = pickle.load(open(f'test_cache/graph_{idx}.pkl', 'rb'))
    except:
        return None

    for lat, lon in zip(lats, longs):
        nearest_nodes.append(ox.distance.nearest_nodes(graph, lat, lon))

    return nearest_nodes

# Prepare data for parallel processing
def prepare_data(df):
    data_to_process = []
    for idx, row in df.iterrows():
        data_to_process.append({
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'i': idx
        })
    return data_to_process

# Main execution
if __name__ == '__main__':

    df = pd.read_csv('data/processed_routes_0_25000.csv')

    df['latitude'] = df['latitude'].apply(eval)
    df['longitude'] = df['longitude'].apply(eval)
    df = df.iloc[0:9360, :]
    # Assuming df is already loaded
    df['i'] = df.index
    
    # Prepare data for parallel processing
    data_to_process = prepare_data(df)
    
    # Set up multiprocessing pool with number of available CPUs
    num_processes = cpu_count()  # You can adjust this if needed
    
    # Create a processing pool and apply the function in parallel
    with Pool(processes=num_processes) as pool:
        # Process with progress bar
        results = list(tqdm(
            pool.imap(process_route, data_to_process),
            total=len(data_to_process),
            desc="Processing routes"
        ))
    
    # Update the DataFrame with results
    df['nearest_nodes'] = results

    df.to_csv('data/processed_for_graph.csv')