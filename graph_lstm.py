import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import osmnx as ox
import networkx as nx
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
from tqdm import tqdm
import functools
import multiprocessing
import traceback
from utils.preprocessing import get_route_distance
import matplotlib.pyplot as plt
import datetime

def get_datetime_string():
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")
from torch_geometric.data import Data

def create_empty_graph_data(node_feature_dim=6, edge_feature_dim=8):
    """Create a standardized empty graph data object"""
    empty_x = torch.zeros((1, node_feature_dim), dtype=torch.float32)
    empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
    empty_edge_attr = torch.zeros((0, edge_feature_dim), dtype=torch.float32)
    return Data(x=empty_x, edge_index=empty_edge_index, edge_attr=empty_edge_attr)

def total_route_distance(route_xy):
    """Calculate the total distance of a route"""
    if len(route_xy) < 2:
        return 0.0
    dist = 0.0
    for i in range(len(route_xy) - 1):
        dist += np.sqrt((route_xy[i+1][0] - route_xy[i][0])**2 + 
                        (route_xy[i+1][1] - route_xy[i][1])**2)
    return dist

def lat_lon_to_graph_point(lat, lon, graph):
    """Convert lat/lon to the nearest node in the graph with error handling for missing CRS"""
    try:
        # Try the standard approach first
        node_id = ox.distance.nearest_nodes(graph, lon, lat)
        return node_id
    except KeyError as e:
        if str(e) == "'crs'":
            # The graph doesn't have a CRS defined, we need to add one
            try:
                # Set the standard WGS 84 CRS as default
                graph.graph['crs'] = 'epsg:4326'
                
                # Try again with the CRS defined
                node_id = ox.distance.nearest_nodes(graph, lon, lat)
                return node_id
            except Exception as e2:
                # If that fails, use a fallback approach
                print(f"CRS addition failed: {e2}, using fallback nearest node approach")
                return find_nearest_node_fallback(lat, lon, graph)
        else:
            # Some other KeyError, try the fallback
            print(f"KeyError in nearest_nodes: {e}, using fallback")
            return find_nearest_node_fallback(lat, lon, graph)
    except Exception as e:
        # Any other error, use fallback
        print(f"Error in nearest_nodes: {e}, using fallback")
        return find_nearest_node_fallback(lat, lon, graph)


def find_nearest_node_fallback(lat, lon, graph):
    """Fallback method to find nearest node when osmnx functions fail"""
    if len(graph.nodes()) == 0:
        # No nodes in graph, return None
        return None
        
    # Find nearest node manually
    min_dist = float('inf')
    nearest_node = None
    
    for node, data in graph.nodes(data=True):
        # Check if node has coordinates
        if 'x' in data and 'y' in data:
            node_lon = data['x']
            node_lat = data['y']
        elif 'lon' in data and 'lat' in data:
            node_lon = data['lon']
            node_lat = data['lat']
        else:
            # Skip nodes without coordinates
            continue
            
        # Calculate distance (simple Euclidean distance is sufficient for small areas)
        dist = (node_lon - lon)**2 + (node_lat - lat)**2
        
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    
    return nearest_node

def get_subgraph_around_point(graph, center_node, n_hops=2):
    """Optimized subgraph extraction"""
    # Use NetworkX's ego_graph which is much faster than manual BFS
    subgraph = nx.ego_graph(graph, center_node, radius=n_hops)
    return subgraph
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
from threading import Lock
import pickle
from scipy.spatial import KDTree
import hashlib
import json

# Add this to the RunningRouteDataset class
class RunningRouteDataset(Dataset):
    def __init__(self, csv_path, transform=None, verbose=False, n_hops=2, cache_dir=None, 
                 num_workers=4, parallel_loading=True, proximity_threshold=0.001, 
                 max_workers=None, batch_size=10, num_samples = 1000, device = "mps"):
        """
        Dataset for running routes using OSMnx graph data
        
        Args:
            csv_path: Path to CSV file with metadata and route information
            transform: Transforms to apply (kept for compatibility)
            verbose: Whether to print verbose logs during data loading
            n_hops: Number of hops to include in the subgraph around each point
            cache_dir: Directory to cache OSMnx graph data
            num_workers: Number of parallel workers for graph loading
            parallel_loading: Whether to enable parallel graph loading
            proximity_threshold: Threshold for considering bounds as similar (in degrees)
            max_workers: Maximum number of workers (defaults to num_workers)
            batch_size: Size of batches for parallel processing
        """
        print(f"Loading dataset from {csv_path}...")
        self.data = pd.read_csv(csv_path)
        self.data = self.data[~self.data['nearest_nodes'].isna()]
        if num_samples < len(self.data):
            self.data = self.data.iloc[0:num_samples, :]
        routes = []
        nearest_nodes = []
        for _, row in self.data.iterrows():
            latitudes = eval(row['latitude'])
            longitudes = eval(row['longitude'])
            nearest_nodes.append(eval(row['nearest_nodes']))
            routes.append(list(zip(latitudes, longitudes)))
        self.nearest_nodes = nearest_nodes
        self.route_xy = routes
        self.transform = transform
        self.verbose = verbose
        self.n_hops = n_hops
        self.cache_dir = cache_dir
        self.graph_cache = {}  # Cache for OSMnx graphs
        self.bounds_cache = {}  # Cache to track which bounds correspond to which graph
        self.cache_lock = Lock()  # Lock for thread-safe cache access
        self.num_workers = min(num_workers, multiprocessing.cpu_count())
        self.max_workers = max_workers if max_workers else self.num_workers
        self.parallel_loading = parallel_loading
        self.proximity_threshold = proximity_threshold
        self.batch_size = batch_size
        self.bounds_index = None  # KDTree for efficient bound matching
        self.bounds_mapping = {}  # Mapping from KDTree indices to cache keys
        self.subgraph_mapping = pickle.load(open('data/sub_graph_dict.pkl', 'rb'))
        self.device = device

        
        # Extract bounds for each route if needed
        self.data['bounds'] = [self._calculate_bounds(route) for route in self.route_xy]
        
        # # Preload graphs in parallel if specified and cache directory exists
        # if self.parallel_loading and self.cache_dir:
        #     os.makedirs(self.cache_dir, exist_ok=True)
        #     # First load existing cached graphs
        #     self._load_existing_cache(num_samples)
        #     # Then load any missing graphs in parallel
        #     self._preload_graphs_parallel()

        
            
        # print("Dataset preparation complete!")
        
    def _load_existing_cache(self, num_samples):
        """Load existing cached graphs and build spatial index for proximity lookups"""
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return
            
        print("Loading existing graph cache...")
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        def is_valid_graph_file(f):
            parts = f.split('_')
            if len(parts) < 2 or not parts[0] == 'graph':
                return False
            try:
                index = int(parts[1].split('.')[0])
                return index < num_samples
            except ValueError:
                return False

        cache_files = [f for f in cache_files if is_valid_graph_file(f)]

        cache_files
        
        if not cache_files:
            print("No existing cached graphs found.")
            return
            
        # Try to load the cache metadata if it exists
        metadata_path = os.path.join(self.cache_dir, 'cache_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    cache_metadata = json.load(f)
                    self.bounds_mapping = cache_metadata.get('bounds_mapping', {})
                    
                # Convert string keys back to tuples
                self.bounds_mapping = {tuple(json.loads(k)): v for k, v in self.bounds_mapping.items()}
                
                # Build KDTree from cached bounds
                if self.bounds_mapping:
                    bounds_centers = []
                    bounds_keys = []
                    
                    for bounds, cache_key in self.bounds_mapping.items():
                        # Calculate the center point of these bounds
                        lat_min, lat_max, lon_min, lon_max = bounds
                        center = ((lat_min + lat_max) / 2, (lon_min + lon_max) / 2)
                        bounds_centers.append(center)
                        bounds_keys.append(bounds)
                        
                    if bounds_centers:
                        self.bounds_index = KDTree(bounds_centers)
                        print(f"Built spatial index with {len(bounds_centers)} cached regions")
            except Exception as e:
                print(f"Error loading cache metadata: {e}")
                exit()
                    
        # Load cached graphs
        loaded_count = 0
        for cache_file in tqdm(cache_files, desc="Loading cached graphs"):
            if cache_file == 'cache_metadata.json':
                continue
                
            cache_path = os.path.join(self.cache_dir, cache_file)
            with open(cache_path, 'rb') as f:
                graph = pickle.load(f)
                
            # Extract route_id from filename, handling both formats
            if cache_file.startswith('graph_'):
                # Remove 'graph_' prefix and '.pkl' suffix
                id_part = cache_file[6:-4]

                route_id = int(id_part)  # Convert to integer

                with self.cache_lock:
                    self.graph_cache[route_id] = graph               
                            
                loaded_count += 1
                    
        print(f"Loaded {loaded_count} graphs from cache")
        
    def _bounds_to_key(self, bounds):
        """Convert bounds to a string key for dictionary lookup"""
        lat_min, lat_max, lon_min, lon_max = bounds
        # Round to reduce floating point precision issues
        key = (round(lat_min, 6), round(lat_max, 6), round(lon_min, 6), round(lon_max, 6))
        return key

        
    def _find_similar_bounds(self, bounds):
        """Find cached graph with similar bounds using spatial index"""
        # print(self.bounds_index)
        # print(self.bounds_mapping)
        if self.bounds_index is None or not self.bounds_mapping:
            return None
            
        # Calculate the center of these bounds
        lat_min, lat_max, lon_min, lon_max = bounds
        center = ((lat_min + lat_max) / 2, (lon_min + lon_max) / 2)
        
        # Query the KDTree for nearby bounds centers
        distances, indices = self.bounds_index.query(center, k=5)
        
        # Check if any of the nearby bounds are within our threshold
        for dist, idx in zip(distances, indices):
            if dist <= self.proximity_threshold * 2:  # Double the threshold for center-to-center distance
                # Get the bounds for this index
                nearby_bounds = list(self.bounds_mapping.keys())[idx]
                
                # Verify that the actual bounds overlap sufficiently
                n_lat_min, n_lat_max, n_lon_min, n_lon_max = nearby_bounds
                
                # Check for sufficient overlap
                lat_overlap = min(lat_max, n_lat_max) - max(lat_min, n_lat_min)
                lon_overlap = min(lon_max, n_lon_max) - max(lon_min, n_lon_min)
                
                bounds_width = lat_max - lat_min
                bounds_height = lon_max - lon_min
                
                # If overlap is at least 80% of the area, consider it a match
                if (lat_overlap > 0 and lon_overlap > 0 and 
                    lat_overlap * lon_overlap >= 0.8 * bounds_width * bounds_height):
                    
                    return self.bounds_mapping[nearby_bounds]
                    
        return None
    
    def _preload_graphs_parallel(self):
        """Preload graphs in parallel to speed up training with improved caching"""
        # Identify which routes need their graphs loaded
        to_load = []
        route_ids = self.data['index'].tolist()
        bounds_list = self.data['bounds'].tolist()
        
        print("Identifying routes that need graph loading...")
        for route_id, bounds in tqdm(zip(route_ids, bounds_list), total=len(route_ids), desc="Checking cache"):
            # Skip if already in memory cache
            if route_id in self.graph_cache:
                continue
                
            # Check if we can find similar bounds in our cache
            similar_key = self._find_similar_bounds(bounds)
            if similar_key and similar_key in self.graph_cache:
                # Reuse the cached graph from similar bounds
                with self.cache_lock:
                    self.graph_cache[route_id] = self.graph_cache[similar_key]
                continue
                
            # Check if it exists in the disk cache by route ID
            if self.cache_dir:
                cache_path = os.path.join(self.cache_dir, f"graph_{route_id}.pkl")
                if os.path.exists(cache_path):
                    continue
                
            
            # If we get here, we need to load this graph
            to_load.append((route_id, bounds))
        
        if not to_load:
            print("All graphs already cached, no need for parallel loading")
            return
            
        print(f"Need to load {len(to_load)} out of {len(route_ids)} graphs")
        print(f"Using {self.num_workers} workers for parallel loading")
        
        # Process in batches to reduce memory pressure and improve worker utilization
        batch_size = min(self.batch_size, len(to_load))
        all_batches = [to_load[i:i+batch_size] for i in range(0, len(to_load), batch_size)]
        
        start_time = time.time()
        graphs_loaded = 0
        skipped_count = 0
        
        # Prepare for batch processing
        for batch in tqdm(all_batches, desc="Loading graph batches"):
            # Dynamically adjust number of workers based on batch size
            batch_workers = min(len(batch), self.max_workers)
            
            # Check for too small batches
            if len(batch) < 2:
                # For single tasks, just process directly
                for route_id, bounds in batch:
                    try:
                        graph, rid = self._fetch_graph_worker(route_id, bounds)
                        self._save_graph_to_cache(graph, rid, bounds)
                        graphs_loaded += 1
                    except Exception as e:
                        if self.verbose:
                            print(f"Error processing route {route_id}: {e}")
                continue
                
            # Use ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor(max_workers=batch_workers) as executor:
                # Submit all tasks in this batch
                future_to_route = {}
                for route_id, bounds in batch:
                    # Check if we already have a similar bounds in our cache before submitting
                    similar_key = self._find_similar_bounds(bounds)
                    if similar_key and similar_key in self.graph_cache:
                        # Reuse the cached graph
                        with self.cache_lock:
                            self.graph_cache[route_id] = self.graph_cache[similar_key]
                        skipped_count += 1
                        continue
                        
                    # Submit the task
                    future = executor.submit(self._fetch_graph_worker, route_id, bounds)
                    future_to_route[future] = (route_id, bounds)
                
                # Process results as they complete
                for future in as_completed(future_to_route):
                    route_id, bounds = future_to_route[future]
                    try:
                        # Get the result
                        graph, rid = future.result()
                        self._save_graph_to_cache(graph, rid, bounds)
                        graphs_loaded += 1
                    except Exception as e:
                        if self.verbose:
                            print(f"Error processing route {route_id}: {e}")
        
        # Save cache metadata
        self._save_cache_metadata()
            
        end_time = time.time()
        print(f"Preloaded {graphs_loaded} graphs, skipped {skipped_count} similar in {end_time - start_time:.1f} seconds")
    
    def _save_graph_to_cache(self, graph, route_id, bounds):
        """Save a graph to both memory and disk cache"""
        # Save to memory cache with thread safety
        with self.cache_lock:
            self.graph_cache[route_id] = graph
            
            # Update bounds mapping
            bounds_key = self._bounds_to_key(bounds)
            self.bounds_mapping[bounds_key] = route_id
        
        # Save to disk if cache directory is specified
        if self.cache_dir:
            # Save by route ID
            route_cache_path = os.path.join(self.cache_dir, f"graph_{route_id}.pkl")
            
            # # Also save by bounds hash for similarity lookup
            # bounds_hash = self._get_bounds_hash(bounds)
            # bounds_cache_path = os.path.join(self.cache_dir, f"graph_bounds_{bounds_hash}.pkl")
            
            try:
                # Only save once (either route hasn't been cached or bounds haven't been cached)
                if not os.path.exists(route_cache_path):
                    with open(route_cache_path, 'wb') as f:
                        pickle.dump(graph, f)
                        
                # if not os.path.exists(bounds_cache_path):
                #     with open(bounds_cache_path, 'wb') as f:
                #         pickle.dump(graph, f)
            except Exception as e:
                if self.verbose:
                    print(f"Error saving graph to cache: {e}")
    
    def _save_cache_metadata(self):
        """Save metadata about cached bounds to disk"""
        if not self.cache_dir:
            return
            
        try:
            # Convert tuple keys to strings for JSON serialization
            bounds_mapping_json = {json.dumps(list(k)): v for k, v in self.bounds_mapping.items()}
            
            metadata = {
                'bounds_mapping': bounds_mapping_json,
                'last_updated': time.time()
            }
            
            metadata_path = os.path.join(self.cache_dir, 'cache_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        except:
            pass
    
    @staticmethod
    def _fetch_graph_worker(route_id, bounds):
        """Worker function to fetch a graph (must be static for multiprocessing)"""
        lat_min, lat_max, lon_min, lon_max = bounds
        
        try:
            # Add a small random jitter to avoid exactly the same API requests
            jitter = 0.00001  # Very small jitter, won't affect route quality
            # Only add jitter to the max bounds to ensure the entire area is still covered
            lat_max += np.random.uniform(0, jitter)
            lon_max += np.random.uniform(0, jitter)
            
            # Get graph with specific tags relevant for running
            tags = {
                'highway': [
                    'path', 'footway', 'track', 'pedestrian', 'trail',
                    'residential', 'service', 'unclassified', 'tertiary',
                    'secondary', 'primary', 'cycleway', 'living_street'
                ]
            }
            
            # Add retries with backoff to handle rate limiting
            max_retries = 3
            retry_delay = 1.0  # Start with 1 second delayç
            
            for retry in range(max_retries):
                try:
                    graph = ox.graph_from_bbox(
                       (lon_min, lat_min, lon_max, lat_max),
                        network_type='all',  # Include all types of streets
                        simplify=True,
                    )
                    break  # Success, exit retry loop
                except Exception as e:
                    if retry < max_retries - 1:  # Don't sleep on the last retry
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise  # Re-raise on final retry
            
            # Add edge attributes for path types with custom weights
            # Lower weights for paths preferable for running
            edge_weights = {
                'path': 0.6, 'footway': 0.7, 'track': 0.8, 'pedestrian': 0.7,
                'trail': 0.5, 'cycleway': 0.8, 'living_street': 0.9,
                'residential': 1.0, 'service': 1.2, 'unclassified': 1.3,
                'tertiary': 1.5, 'secondary': 1.8, 'primary': 2.0
            }
            
            for u, v, k, data in graph.edges(keys=True, data=True):
                highway = data.get('highway', '')
                if isinstance(highway, list):
                    highway = highway[0] if highway else ''
                    
                # Set a default weight
                data['weight'] = 1.5
                
                # Set weight based on the type of highway
                if highway in edge_weights:
                    data['weight'] = edge_weights[highway]
            
            return graph, route_id
            
        except Exception as e:
            # Return an empty graph as fallback
            import networkx as nx
            return nx.Graph(), route_id
            
    def get_graph_for_route(self, route_id, bounds):
        """Get or create an OSMnx graph for a route"""
        # First check in-memory cache
        with self.cache_lock:
            if route_id in self.graph_cache:
                return self.graph_cache[route_id]
        
        # Check for a graph with similar bounds 
        similar_key = self._find_similar_bounds(bounds)
        if similar_key:
            with self.cache_lock:
                if similar_key in self.graph_cache:
                    # Clone the graph to avoid any modification issues
                    similar_graph = self.graph_cache[similar_key]
                    # Cache this for future use
                    self.graph_cache[route_id] = similar_graph
                    return similar_graph
                
        # Then check disk cache if available
        if self.cache_dir:
            # Try route ID specific cache first
            cache_path = os.path.join(self.cache_dir, f"graph_{route_id}.pkl")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        graph = pickle.load(f)
                        
                        # Update in-memory cache
                        with self.cache_lock:
                            self.graph_cache[route_id] = graph
                            
                        return graph
                except Exception as e:
                    if self.verbose:
                        print(f"Error loading graph from cache: {e}")
            
            # Try bounds-based cache
            bounds_hash = self._get_bounds_hash(bounds)
            bounds_cache_path = os.path.join(self.cache_dir, f"graph_bounds_{bounds_hash}.pkl")
            if os.path.exists(bounds_cache_path):
                try:
                    with open(bounds_cache_path, 'rb') as f:
                        graph = pickle.load(f)
                        
                        # Update in-memory cache
                        with self.cache_lock:
                            self.graph_cache[route_id] = graph
                            # Update bounds mapping
                            bounds_key = self._bounds_to_key(bounds)
                            self.bounds_mapping[bounds_key] = route_id
                            
                        return graph
                except Exception as e:
                    if self.verbose:
                        print(f"Error loading graph from bounds cache: {e}")
        
        # If not found in cache, fetch directly (fallback to sequential for single routes)
        if self.verbose:
            print(f"Fetching graph for route {route_id} with bounds: {bounds}")
            
        # Use the worker function directly
        graph, _ = self._fetch_graph_worker(route_id, bounds)
        
        # Save to cache
        self._save_graph_to_cache(graph, route_id, bounds)
                
        return graph


    def _calculate_bounds(self, route_xy):
        """Calculate the bounds of a route to use for OSMnx queries"""
        # print(route_xy)
        # print(type(route_xy))
        lat_min = min(point[0] for point in route_xy) - 0.01  # Add a small buffer
        lat_max = max(point[0] for point in route_xy) + 0.01
        lon_min = min(point[1] for point in route_xy) - 0.01
        lon_max = max(point[1] for point in route_xy) + 0.01
        return (lat_min, lat_max, lon_min, lon_max)

    
    def __len__(self):
        return len(self.data)
    
    def extract_node_features(self, graph, node_id):
        """Extract features for a node in the graph"""
        # Basic node features: degree, is_intersection
        try:
            node_data = graph.nodes[node_id]
            degree = graph.degree(node_id)
            is_intersection = 1.0 if degree > 2 else 0.0
            
            # Extract more features if available
            elevation = node_data.get('elevation', 0.0)
            traffic_signals = 1.0 if node_data.get('highway') == 'traffic_signals' else 0.0
            crossing = 1.0 if node_data.get('highway') == 'crossing' else 0.0
            
            # Combine features
            features = [
                degree / 10.0,  # Normalize degree
                is_intersection,
                elevation / 100.0 if elevation else 0.0,  # Normalize elevation
                traffic_signals,
                crossing
            ]
            
            return torch.tensor(features, dtype=torch.float32)
        except:
            # Return a default feature vector if node doesn't exist
            return torch.zeros(5, dtype=torch.float32)
    
    def extract_edge_features(self, graph, u, v):
        """Extract features for an edge in the graph"""
        try:
            data = graph.get_edge_data(u, v)
            if data is None:
                return torch.zeros(8, dtype=torch.float32)
                
            # Choose the first edge if there are multiple edges
            if isinstance(data, dict) and 0 in data:
                data = data[0]
                
            # Extract highway type
            highway = data.get('highway', '')
            if isinstance(highway, list):
                highway = highway[0] if highway else ''
                
            # One-hot encoding for common highway types
            is_footway = 1.0 if highway == 'footway' else 0.0
            is_path = 1.0 if highway == 'path' else 0.0
            is_track = 1.0 if highway == 'track' else 0.0
            is_trail = 1.0 if highway == 'trail' else 0.0
            is_residential = 1.0 if highway == 'residential' else 0.0
            
            # Other features
            length = data.get('length', 0.0) / 1000.0  # Normalize to km
            weight = data.get('weight', 1.5)
            is_oneway = 1.0 if data.get('oneway', False) else 0.0
            
            features = [
                is_footway,
                is_path,
                is_track,
                is_trail,
                is_residential,
                length,
                weight,
                is_oneway
            ]
            
            return torch.tensor(features, dtype=torch.float32)
        except:
            # Return a default feature vector if edge doesn't exist
            return torch.zeros(8, dtype=torch.float32)
    
    
    def __getitem__(self, idx):
        # Get the row from the dataframe
        row = self.data.iloc[idx]
        
        # Get route coordinates
        route_xy = self.route_xy[idx]
        route_tensor = torch.tensor(route_xy, dtype=torch.float32)
        
        # Get conditional parameters
        distance = torch.tensor([row['distance']], dtype=torch.float32)

        # Extract start and end points
        start_point = torch.tensor(route_xy[0], dtype=torch.float32)
        end_point = torch.tensor(route_xy[-1], dtype=torch.float32)
        
        # Get latitude and longitude
        route_lat = eval(row['latitude']) if 'latitude' in row else np.array([p[0] for p in route_xy])
        route_lon = eval(row['longitude']) if 'longitude' in row else np.array([p[1] for p in route_xy])
        
        # Get or create graph for this route
        route_id = row['i']

        sub_graphs = [self.subgraph_mapping[node] for node in eval(row['nearest_nodes'])[:-1]]
        graph = self.get_graph_for_route(route_id, row['bounds'])

        # closest_nodes = [ox.distance.nearest_nodes(graph, lat, lon) for lat, lon in zip(route_lat, route_lon)]

        # nodes = set(closest_nodes)

        # for node in nodes:
        #     if self.subgraph_mapping.get(node, 0) != 0:
        #         nodes.remove(node)

        # nodes = list(nodes)        
        # tasks = []

            # Second, create processing tasks with unique node information
        # for i, node in enumerate(nodes):
        #     # Create a unique index for this task
        #     tasks.append((idx, node))
        
        # Process in parallel
        # processed_graphs = [None] * len(tasks)
        
        # Use fewer workers if there are few tasks
        # actual_workers = min(self.num_workers, max(1, len(tasks)))
        
        # with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        #     def worker(args):
        #         idx, node_id = args
        #         # Call the cached function
        #         graph_data = process_node(node_id, graph)
        #         self.subgraph_mapping[node_id] = graph_data
        #         return idx, graph_data
            
        #     for i, (task_idx, graph_data) in enumerate(executor.map(worker, tasks)):
        #         processed_graphs[i] = graph_data 

        # sub_graphs = [self.subgraph_mapping[i].to(self.device) for i in closest_nodes]
        
        # Create input-target pairs for sequence learning
        # Input: all coordinates except the last one
        # Target: all coordinates except the first one
        input_seq = route_tensor[:-1]
        target_seq = route_tensor[1:]
        
        # Calculate steps remaining for each position in the sequence
        seq_length = len(route_xy)
        steps_remaining = torch.arange(seq_length-1, 0, -1).float()
        
        # Get lat/lon for input sequence
        input_lat = route_lat[:-1] if isinstance(route_lat, np.ndarray) else np.array(route_lat)[:-1]
        input_lon = route_lon[:-1] if isinstance(route_lon, np.ndarray) else np.array(route_lon)[:-1]
        
        # For returning processed graph data during training, we'll process all points later
        # Just return the necessary information to create the graphs
        
        return {
            'route': route_tensor,
            'input_seq': input_seq,
            'target_seq': target_seq,
            'conditions': torch.cat([distance, start_point, end_point], dim=0),
            'seq_length': seq_length,
            'steps_remaining': steps_remaining,
            'route_id': route_id,
            'input_lat': input_lat,
            'input_lon': input_lon,
            'subgraph': sub_graphs,
            'graph' : graph,
            'nearest_nodes': eval(row['nearest_nodes'])[:-1],
            'bounds': row['bounds']  # Return the full graph
        }
    
def create_consistent_graph_data(subgraph, center_node, node_feature_dim=6, edge_feature_dim=8):
    """
    Create PyTorch Geometric data with guaranteed edge-node consistency
    
    Args:
        subgraph: NetworkX subgraph
        center_node: Center node ID
        node_feature_dim: Dimension of node features
        edge_feature_dim: Dimension of edge features
        
    Returns:
        graph_data: PyTorch Geometric Data with consistent dimensions
    """
    # If empty subgraph, return standard empty graph
    if len(subgraph.nodes) == 0:
        empty_x = torch.zeros((1, node_feature_dim), dtype=torch.float32)
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        empty_edge_attr = torch.zeros((0, edge_feature_dim), dtype=torch.float32)
        return Data(x=empty_x, edge_index=empty_edge_index, edge_attr=empty_edge_attr)
        
    # First collect all edges and their attributes in matched lists
    edge_indices = []
    edge_attributes = []
    
    # Create node mapping to ensure consecutive node indices
    node_map = {node: idx for idx, node in enumerate(subgraph.nodes())}
    
    # Process nodes
    x = torch.zeros((len(node_map), node_feature_dim), dtype=torch.float32)
    # Set node features here...
    
    # Mark center node with special feature
    if center_node in node_map:
        center_idx = node_map[center_node]
        # Set center node indicator (example: last feature)
        x[center_idx, -1] = 1.0
    
    # Process edges
    for u, v, data in subgraph.edges(data=True):
        if u in node_map and v in node_map:
            # Add edge in both directions for undirected graph
            edge_indices.append([node_map[u], node_map[v]])
            edge_indices.append([node_map[v], node_map[u]])
            
            # Create edge attributes from data
            edge_attr = torch.zeros(edge_feature_dim, dtype=torch.float32)
            # Set edge features here based on data...
            
            # Add same attributes for both directions
            edge_attributes.append(edge_attr)
            edge_attributes.append(edge_attr.clone())
    
    # Convert to tensor format
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()  # Transpose to [2, num_edges]
        edge_attr = torch.stack(edge_attributes)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, edge_feature_dim), dtype=torch.float32)
    
    # One final check to ensure dimensions match
    assert edge_index.shape[1] == edge_attr.shape[0], f"Edge dimensions mismatch: {edge_index.shape[1]} vs {edge_attr.shape[0]}"
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class GraphEncoder(nn.Module):
    """Encodes local map subgraphs around the current position with tensor size safety checks"""
    def __init__(self, node_feature_dim=6, edge_feature_dim=8, hidden_dim=64, output_dim=256):
        super(GraphEncoder, self).__init__()
        
        # Node embedding layers
        self.node_emb = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Edge embedding layers
        self.edge_emb = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Graph convolution layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Output layer
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, data):
        """
        Forward pass through the graph encoder with tensor size safety checks
        """
        # Safety check for empty data
        if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
            print("Warning: Graph data missing nodes or edges")
            batch_size = 1
            if hasattr(data, 'batch') and data.batch is not None and data.batch.numel() > 0:
                batch_size = data.batch.max().item() + 1
            return torch.zeros(batch_size, self.out_layer[0].out_features, device=data.x.device if hasattr(data, 'x') else 'cpu')
        
        # Embed node features
        x = self.node_emb(data.x)
        
        # Check if there are edges to process
        if data.edge_index.shape[1] == 0:
            # No edges, just pool node features directly
            batch_size = data.batch.max().item() + 1 if data.batch.numel() > 0 else 1
            output = []
            for i in range(batch_size):
                mask = (data.batch == i)
                if mask.sum() > 0:
                    output.append(x[mask].mean(dim=0))
                else:
                    output.append(torch.zeros(x.size(1), device=x.device))
            output = torch.stack(output)
            return self.out_layer(output) 
        
        # Process edge attributes to create edge weights (1D tensor)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            try:
                # Embed edge features
                edge_attr_embedded = self.edge_emb(data.edge_attr)
                
                # Convert 2D edge attributes to 1D edge weights
                # Option 1: Use mean of each edge feature vector
                edge_weight = edge_attr_embedded.mean(dim=1)
                
                # Option 2 (alternative): Use sum instead of mean
                # edge_weight = edge_attr_embedded.sum(dim=1)
                
            except Exception as e:
                print(f"Error processing edge attributes: {e}")
                # Create default edge weights (all ones)
                edge_weight = torch.ones(data.edge_index.shape[1], device=x.device)
        else:
            # Create default edge weights (all ones)
            edge_weight = torch.ones(data.edge_index.shape[1], device=x.device)
        
        # Apply graph convolutions with the 1D edge weights
        try:
            x = F.relu(self.conv1(x, data.edge_index, edge_weight))
            x = F.dropout(x, p=0.2, training=self.training)
            
            x = F.relu(self.conv2(x, data.edge_index, edge_weight))
            x = F.dropout(x, p=0.2, training=self.training)
            
            x = F.relu(self.conv3(x, data.edge_index, edge_weight))
        except Exception as e:
            print(f"Error in graph convolutions: {e}")
            # If convolutions fail, we can still return a result based on node features
        
        # Global pooling
        batch_size = data.batch.max().item() + 1 if data.batch.numel() > 0 else 1
        output = []
        
        for i in range(batch_size):
            mask = (data.batch == i)
            graph_nodes = x[mask]
            
            if graph_nodes.size(0) > 0:
                graph_feat = graph_nodes.mean(dim=0)
            else:
                graph_feat = torch.zeros(x.size(1), device=x.device)
                
            output.append(graph_feat)
        
        output = torch.stack(output)
        return self.out_layer(output)

class ConditionEncoder(nn.Module):
    """
    Encodes the conditioning information
    (distance, start, end points, steps remaining)
    """
    def __init__(self, input_dim=6, output_dim=64):
        super(ConditionEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

# Define this outside of any class, at the module level
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
    
class ImprovedRoutePredictor(nn.Module):
    """LSTM model for predicting the next coordinate based on previous coordinates
    and graph features"""
    def __init__(self, graph_feature_dim=256, condition_dim=64, hidden_dim=256, num_layers=2):
        super(ImprovedRoutePredictor, self).__init__()
        
        # Coordinate embedder - convert raw (x,y) into a richer representation
        self.coord_embedder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=32,  # embedded coordinate dimension
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Context integration layer - combines LSTM output with graph and condition features
        self.context_layer = nn.Sequential(
            nn.Linear(hidden_dim + graph_feature_dim + condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Final coordinate prediction layer
        self.output_layer = nn.Linear(64, 2)
    
    def forward(self, coords, graph_features, condition_features, hidden=None):
        """
        Forward pass through the route predictor
        
        Args:
            coords: Tensor of shape [batch_size, seq_len, 2] with input coordinates
            graph_features: Tensor of shape [batch_size, seq_len, graph_feature_dim]
            condition_features: Tensor of shape [batch_size, seq_len, condition_dim]
            hidden: Initial hidden state (optional)
            
        Returns:
            next_coords: Predicted next coordinates
            hidden: Updated hidden state
        """
        batch_size, seq_len, _ = coords.shape
        
        # Embed coordinates
        embedded_coords = self.coord_embedder(coords)
        
        # Process sequence with LSTM
        lstm_out, hidden = self.lstm(embedded_coords, hidden)
        
        # Combine LSTM output with context (graph features and conditions)
        combined = torch.cat([lstm_out, graph_features, condition_features], dim=2)
        context_integrated = self.context_layer(combined)
        
        # Predict next coordinates
        next_coords = self.output_layer(context_integrated)
        
        return next_coords, hidden


        
class GraphAwareRouteGenerator(nn.Module):
    """Enhanced model with graph awareness and steps remaining information"""
    def __init__(self, graph_feature_dim=256, condition_dim=64, hidden_dim=256, 
                num_layers=2, n_hops=2):
        super(GraphAwareRouteGenerator, self).__init__()
        
        self.graph_feature_dim = graph_feature_dim
        self.n_hops = n_hops
        
        # Graph encoder instead of image encoder
        self.graph_encoder = GraphEncoder(
            node_feature_dim=4,  # 5 node features + 1 for center node indicator
            edge_feature_dim=5,
            hidden_dim=128,
            output_dim=graph_feature_dim
        )
        
        # Enhanced condition encoder that includes steps remaining
        self.condition_encoder = ConditionEncoder(
            input_dim=6,  # distance, start_x, start_y, end_x, end_y, steps_remaining
            output_dim=condition_dim
        )
        
        # Improved route predictor
        self.route_predictor = ImprovedRoutePredictor(
            graph_feature_dim=graph_feature_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

    def prepare_graph_batch_simple(self, graphs, lat_coords, lon_coords, device):
        """
        Simplified graph batch preparation that processes each graph once
        without subgraph extraction or error handling
        """
        batch_size = len(lat_coords)
        processed_graphs = []
        
        for b in range(batch_size):
            graph = graphs[b]

            for lat, lon in zip(lat_coords[b], lon_coords[b]):
                # Get the center node ID
                node_id = ox.distance.nearest_nodes(graph, lon, lat)
                
                # Create subgraph - note: node_id is used directly, not graph.nodes[node_id]
                subgraph = nx.ego_graph(graph, node_id, radius=2)

                # Create single node feature tensor for all nodes
                num_nodes = len(subgraph.nodes())
                x = torch.zeros((num_nodes, 4), dtype=torch.float32)
                
                # Create node mapping for the SUBGRAPH (not the original graph)
                node_map = {node: idx for idx, node in enumerate(subgraph.nodes())}
                
                # Add basic node features
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
                    # Use the subgraph node mapping
                    src_idx = node_map[u]
                    dst_idx = node_map[v]
                    
                    # Add bidirectional edges
                    edge_indices.append([src_idx, dst_idx])
                    edge_indices.append([dst_idx, src_idx])
                    
                    # Create edge attributes
                    edge_attr = torch.zeros(5, dtype=torch.float32)
                    
                    # Add real edge features
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
                        
                    # Add same attributes for both directions
                    edge_attrs.append(edge_attr)
                    edge_attrs.append(edge_attr.clone())
            
                # Convert to tensors
                if edge_indices:
                    edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
                    edge_attr = torch.stack(edge_attrs)
                else:
                    # Handle empty case
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                    edge_attr = torch.zeros((0, 5), dtype=torch.float32)
                
                graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                processed_graphs.append(graph_data)
        
        # Create batch
        graph_batch = Batch.from_data_list(processed_graphs).to(device)
        return graph_batch

    def prepare_graph_batch_cached_parallel(self, graphs, lat_coords, lon_coords, device, num_workers=50):
        """Parallel implementation with caching of common nodes"""
        batch_size = len(lat_coords)
        tasks = []
        
        # First, find all nearest nodes to avoid redundant calculations
        node_mapping = {}  # Maps (graph_idx, lat, lon) -> (node_id, graph_id)
        graph_id_map = {}  # Store a unique ID for each graph object to use in caching
        
        for b in range(batch_size):
            graph = graphs[b]
            
            # Create a unique identifier for this graph
            graph_id = id(graph)  # Using object id as identifier
            graph_id_map[graph_id] = graph
            
            # First, find all nearest nodes for this graph in one batch if possible
            coords = [(lat, lon) for lat, lon in zip(lat_coords[b], lon_coords[b])]
            
            # You could potentially do this in parallel too, but often OSMnx has
            # batch operations that are more efficient
            for i, (lat, lon) in enumerate(coords):
                node_id = ox.distance.nearest_nodes(graph, lon, lat)
                node_mapping[(b, i)] = (node_id, graph_id)
        
        # Second, create processing tasks with unique node information
        for (b, i), (node_id, graph_id) in node_mapping.items():
            # Create a unique index for this task
            idx = len(tasks)
            tasks.append((idx, node_id, graph_id, b, i))
        
        # Process in parallel
        processed_graphs = [None] * len(tasks)
        
        # Use fewer workers if there are few tasks
        actual_workers = min(num_workers, max(1, len(tasks)))
        
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            def worker(args):
                idx, node_id, graph_id, b, i = args
                # Call the cached function
                graph_data = process_node_with_cache(node_id, graph_id, graphs[b])
                return idx, graph_data
            
            for idx, graph_data in executor.map(worker, tasks):
                processed_graphs[idx] = graph_data
        
        # Move all graphs to the right device
        processed_graphs = [g.to(device) for g in processed_graphs]
        
        # Create batch
        graph_batch = Batch.from_data_list(processed_graphs)
        return graph_batch
    
    def prepare_graph_batch_simple_full_map(self, graphs, lat_coords, lon_coords, device):
        """
        Simplified graph batch preparation that processes each graph once
        without subgraph extraction or error handling
        """
        batch_size = len(lat_coords)
        processed_graphs = []
        
        for b in range(batch_size):
            graph = graphs[b]
            seq_len = len(lat_coords[b])
            
            # Create single node feature tensor for all nodes
            num_nodes = len(graph.nodes())
            x = torch.zeros((num_nodes, 6), dtype=torch.float32)
            
            # Add basic node features (could add real features here)
            for i, (node_id, node_data) in enumerate(graph.nodes(data=True)):
                if 'pos' in node_data:
                    x[i, 0:2] = torch.tensor(node_data['pos'], dtype=torch.float32)
                if 'elevation' in node_data:
                    x[i, 2] = node_data['elevation']
                if 'highway' in node_data:  # One-hot encoding for road type
                    x[i, 3] = 1.0
            
            # Create node mapping
            node_map = {node: idx for idx, node in enumerate(graph.nodes())}
            
            # Process edges to create edge_index and edge_attr tensors
            num_edges = len(graph.edges())
            edge_indices = []
            edge_attrs = []
            
            for u, v, data in graph.edges(data=True):
                # Convert to indices in our node mapping
                src_idx = node_map[u]
                dst_idx = node_map[v]
                
                # Add bidirectional edges
                edge_indices.append([src_idx, dst_idx])
                edge_indices.append([dst_idx, src_idx])
                
                # Create edge attributes
                edge_attr = torch.zeros(8, dtype=torch.float32)
                
                # Add real edge features if available
                if 'length' in data:
                    edge_attr[0] = data['length']
                if 'grade' in data:
                    edge_attr[1] = data['grade']
                if 'highway' in data:
                    edge_attr[2] = 1.0
                if 'oneway' in data and data['oneway']:
                    edge_attr[3] = 1.0
                    
                # Add same attributes for both directions
                edge_attrs.append(edge_attr)
                edge_attrs.append(edge_attr.clone())
            
            # Convert to tensors
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
            edge_attr = torch.stack(edge_attrs)
            
            # Create one graph object per sequence position
            for s in range(seq_len):
                graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                processed_graphs.append(graph_data)
        
        # Create batch
        graph_batch = Batch.from_data_list(processed_graphs).to(device)
        return graph_batch
    
    def prepare_graph_batch(self, graphs, lat_coords, lon_coords, device):
        """Optimized graph batch creation with multi-level caching"""
        batch_size = len(lat_coords)
        subgraphs = []
        
        # Create cache if it doesn't exist
        if not hasattr(self, 'subgraph_cache'):
            self.subgraph_cache = {}
        
        for b in range(batch_size):
            graph = graphs[b]
            if graph is None or not isinstance(graph, nx.Graph):
                # Add an empty graph and continue
                subgraphs.append(create_empty_graph_data(6, 8).to(device))
                continue
                
            seq_len = len(lat_coords[b]) if b < len(lat_coords) else 0
            
            for s in range(seq_len):
                # Create a cache key based on coordinates (rounded to reduce variations)
                lat = round(lat_coords[b][s], 5)
                lon = round(lon_coords[b][s], 5)
                cache_key = f"{id(graph)}_{lat}_{lon}"
                
                # Check if we've already processed this location
                if cache_key in self.subgraph_cache:
                    subgraphs.append(self.subgraph_cache[cache_key].clone().to(device))
                    continue
                    
                try:
                    # Process the subgraph as before
                    center_node = lat_lon_to_graph_point(lat_coords[b][s], lon_coords[b][s], graph)
                    subgraph = get_subgraph_around_point(graph, center_node, n_hops=self.n_hops)
                    graph_data = create_consistent_graph_data(subgraph, center_node)
                    
                    # Store in cache (on CPU to save GPU memory)
                    self.subgraph_cache[cache_key] = graph_data.to('cpu')
                    
                    # Add to current batch
                    subgraphs.append(graph_data.to(device))
                except Exception as e:
                    # Add an empty graph on error
                    subgraphs.append(create_empty_graph_data(6, 8).to(device))
                    
        # Clear cache if it gets too large (adjust threshold as needed)
        if len(self.subgraph_cache) > 1000000:
            self.subgraph_cache.clear()
        
        # Create batch with minimal error checking (we trust our cache)
        return Batch.from_data_list(subgraphs)

    
    def prepare_batch_data(self, batch_data, device):
        """Prepare data for the forward pass during training"""
        # Unpack batch data
        input_seq = batch_data['input_seq'].to(device)
        input_lat = batch_data['input_lat']
        input_lon = batch_data['input_lon']
        conditions = batch_data['conditions'].to(device)
        steps_remaining = batch_data['steps_remaining'].to(device)
        subgraphs = batch_data['subgraph']

        subgraphs = [sub for graph in subgraphs for sub in graph]

        graph_batch = Batch.from_data_list(subgraphs).to(device)

        # Encode graph features
        graph_features = self.graph_encoder(graph_batch)
        
        # Reshape to [batch_size, seq_len, feature_dim]
        batch_size, seq_len = input_seq.shape[:2]
        graph_features = graph_features.view(batch_size, seq_len, -1)
        
        # Enhance conditions with steps remaining
        enhanced_conditions = []
        for b in range(batch_size):
            # Combine base conditions with steps remaining for each position
            base_cond = conditions[b].unsqueeze(0).expand(seq_len, -1)  # [seq_len, 5]
            steps = steps_remaining[b, :seq_len].unsqueeze(1)  # [seq_len, 1]
            combined = torch.cat([base_cond, steps], dim=1)  # [seq_len, 6]
            enhanced_conditions.append(combined)
        
        enhanced_conditions = torch.stack(enhanced_conditions).to(device)  # [batch_size, seq_len, 6]
        
        # Encode enhanced conditions
        condition_features = self.condition_encoder(enhanced_conditions)
        
        return input_seq, graph_features, condition_features
    
    def forward(self, batch_data):
        """
        Forward pass through the model for training
        
        Args:
            batch_data: Dictionary containing all the required data
            
        Returns:
            predicted_coords: Predicted next coordinates for each input position
        """
        device = next(self.parameters()).device
        input_seq, graph_features, condition_features = self.prepare_batch_data(batch_data, device)
        
        # Predict next coordinates
        predicted_coords, _ = self.route_predictor(
            input_seq, 
            graph_features, 
            condition_features
        )
        
        return predicted_coords
    
    def generate_route(self, graph, start_lat, start_lon, end_lat, end_lon, 
                       distance, nearest_nodes, subgraphs, max_length=500, device='cpu'):
        """
        Generate a complete route autoregressively
        
        Args:
            graph: NetworkX graph for the area
            start_lat, start_lon: Starting coordinates
            end_lat, end_lon: Ending coordinates
            distance: Desired distance of the route
            max_length: Maximum route length to generate
            device: Device to place tensors on
            
        Returns:
            generated_route: The generated route coordinates
        """
        with torch.no_grad():
            # Convert start and end points to graph coordinates
            # start_node = lat_lon_to_graph_point(start_lat, start_lon, graph)
            # end_node = lat_lon_to_graph_point(end_lat, end_lon, graph)
            
            # Initialize the route with the start point
            start_point = torch.tensor([[start_lat, start_lon]], dtype=torch.float32, device=device)
            generated_route = [start_point]
            
            # Conditions tensor: [distance, start_lat, start_lon, end_lat, end_lon]
            conditions = torch.tensor(
                [[distance, start_lat, start_lon, end_lat, end_lon]], 
                dtype=torch.float32, 
                device=device
            )
            
            # Hidden state for LSTM
            hidden = None
            
            # Generate points one by one autoregressively
            current_point = start_point
            current_lat_lon = np.array([[start_lat, start_lon]])
            
            for step in range(max_length - 1):
                # Calculate steps remaining
                steps_remaining = torch.tensor(
                    [[max_length - step - 1]], 
                    dtype=torch.float32, 
                    device=device
                )
                
                # Get the subgraph around current point
                # center_node = lat_lon_to_graph_point(current_lat_lon[0, 0], current_lat_lon[0, 1], graph)
                # subgraph = get_subgraph_around_point(graph, center_node, n_hops=self.n_hops)
                
                graph_data = subgraphs[step].to(device)
                
                # Encode the graph
                graph_features = self.graph_encoder(Batch.from_data_list([graph_data]))
                graph_features = graph_features.unsqueeze(1)  # [1, 1, feature_dim]
                
                # Enhance conditions with steps remaining
                enhanced_conditions = torch.cat([
                    conditions, 
                    steps_remaining
                ], dim=1).unsqueeze(1)  # [1, 1, 6]
                
                # Encode conditions
                condition_features = self.condition_encoder(enhanced_conditions)
                
                # Predict next coordinate
                next_coord_pred, hidden = self.route_predictor(
                    current_point.unsqueeze(1), 
                    graph_features, 
                    condition_features, 
                    hidden
                )
                
                # Get the predicted coordinate
                next_point = next_coord_pred.squeeze(1)
                
                # Add to the generated route
                generated_route.append(next_point)
                
                # Update current point for next iteration
                current_point = next_point
                current_lat_lon = current_point.cpu().numpy()
                
                # Check if we're close enough to the end point
                end_points = torch.tensor([[end_lat, end_lon]], device=device)
                distances_to_end = torch.norm(current_point - end_points, dim=1)
                
                # If all routes are close to their end points, we can stop early
                if torch.all(distances_to_end < 0.0001):  # Small threshold for lat/lon
                    break
            
            # Concatenate all coordinates
            full_route = torch.cat(generated_route, dim=0)
            
        return full_route
def create_padded_batch(batch_data):
    """
    Create a padded batch from a list of samples with varying sequence lengths
    
    Args:
        batch_data: List of dictionaries, each containing sample data
        
    Returns:
        batch_dict: Dictionary with batched and padded tensors
    """
    # Get the maximum sequence length in this batch
    max_seq_len = max([data['seq_length'] for data in batch_data])
    
    # Extract and batch data
    batch_size = len(batch_data)
    batch_dict = {
        'route_id': [data['route_id'] for data in batch_data],
        'seq_length': torch.tensor([data['seq_length'] for data in batch_data]),
        'conditions': torch.stack([data['conditions'] for data in batch_data])
    }
    
    # Pad input and target sequences
    padded_input_seq = torch.zeros(batch_size, max_seq_len-1, 2)
    padded_target_seq = torch.zeros(batch_size, max_seq_len-1, 2)
    padded_steps_remaining = torch.zeros(batch_size, max_seq_len-1)


    
    # Input latitude and longitude for graph processing
    input_lat = []
    input_lon = []
    subgraphs = []
    
    for i, data in enumerate(batch_data):
        seq_len = data['seq_length']
        # Only pad up to max_seq_len-1 because we're dealing with input/target pairs
        padded_input_seq[i, :seq_len-1] = data['input_seq']
        padded_target_seq[i, :seq_len-1] = data['target_seq']
        padded_steps_remaining[i, :seq_len-1] = data['steps_remaining']
        
        input_lat.append(data['input_lat'])
        input_lon.append(data['input_lon'])
        subgraphs.append(data['subgraph'])
        # graph.append(data['graph'])
    
    batch_dict.update({
        'input_seq': padded_input_seq,
        'target_seq': padded_target_seq,
        'steps_remaining': padded_steps_remaining,
        'input_lat': input_lat,
        'input_lon': input_lon,
        'subgraph': subgraphs  # Keep all graphs in the batch
    })
    
    return batch_dict


class RouteModelTrainer:
    """Class to handle training and evaluation of the route model"""
    def __init__(self, model, train_loader, val_loader=None, 
                 lr=0.001, weight_decay=1e-5, device='mps', 
                 disable_graph=False, verbose=True, 
                 tensorboard_log_dir=None):
        """
        Initialize the trainer
        
        Args:
            model: The model to train
            train_loader: DataLoader for the training set
            val_loader: DataLoader for the validation set (optional)
            lr: Learning rate
            weight_decay: Weight decay for regularization
            device: Device to use for training
            disable_graph: Whether to disable graph encoding
            verbose: Whether to print verbose output
            tensorboard_log_dir: Directory for TensorBoard logs (if None, TensorBoard logging is disabled)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model.to(device)
        self.disable_graph = disable_graph
        self.verbose = verbose
        
        # TensorBoard setup
        self.tensorboard_writer = None
        if tensorboard_log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_log_dir)
                print(f"TensorBoard logging enabled. Log directory: {tensorboard_log_dir}")
            except ImportError:
                print("Warning: TensorBoard not available. Install with 'pip install tensorboard'")
        
        # Print training configuration
        print(f"Training configuration:")
        print(f"  - Device: {device}")
        print(f"  - Learning rate: {lr}")
        print(f"  - Weight decay: {weight_decay}")
        print(f"  - Graph encoding: {'Disabled' if disable_graph else 'Enabled'}")
        print(f"  - TensorBoard: {'Enabled' if self.tensorboard_writer else 'Disabled'}")
        
        # Define optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Define learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )

    
    def compute_loss(self, predictions, targets):
        """
        Compute the loss function
        
        Args:
            predictions: Predicted coordinates [batch_size, seq_len, 2]
            targets: Target coordinates [batch_size, seq_len, 2]
            
        Returns:
            loss: Combined loss value
            loss_components: Dictionary with individual loss components
        """
        # MSE loss for coordinate prediction
        mse_loss = F.mse_loss(predictions, targets)
        
        # Initialize direction loss
        direction_loss = 0.0
        
        # Additional term: encourage smooth routes by penalizing sudden changes in direction
        # Get vectors between consecutive predicted points
        if predictions.size(1) > 1:  # Only compute direction loss if we have at least 2 points
            vectors = predictions[:, 1:] - predictions[:, :-1]
            # Get vectors between consecutive target points
            target_vectors = targets[:, 1:] - targets[:, :-1]
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            
            # Normalize vectors to focus on direction, not magnitude
            vectors_norm = F.normalize(vectors + epsilon, dim=2)
            target_vectors_norm = F.normalize(target_vectors + epsilon, dim=2)
            
            # Calculate cosine similarity (higher means more similar direction)
            # We want high similarity, so we take 1 - similarity as our loss
            direction_loss = 1 - torch.mean(
                torch.sum(vectors_norm * target_vectors_norm, dim=2)
            )
            
            # Combine the losses
            # You can adjust these weights based on your priorities
            combined_loss = mse_loss + .2 * direction_loss
        else:
            combined_loss = mse_loss
        
        # Return the combined loss and individual components for logging
        loss_components = {
            'mse_loss': mse_loss.item(),
            'direction_loss': direction_loss if isinstance(direction_loss, float) else direction_loss.item()
        }
        
        return combined_loss, loss_components
    
    def train_epoch(self, epoch):
        """
        Train the model for one epoch with tensor shape checking
        
        Args:
            epoch: Current epoch number (for TensorBoard logging)
            
        Returns:
            avg_loss: Average loss for the epoch
            avg_loss_components: Average loss components for the epoch
        """
        self.model.train()
        total_loss = 0
        total_loss_components = {'mse_loss': 0, 'direction_loss': 0}
        n_batches = 0
        error_count = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(self.train_loader)
        for batch_idx, batch_data in enumerate(pbar):
            try:
                # Create padded batch
                batch = create_padded_batch(batch_data)
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass with graph encoding disabled if specified
                predicted_coords = self.model(batch)
                
                # Check shapes before loss computation
                if isinstance(predicted_coords, torch.Tensor) and isinstance(batch['target_seq'], torch.Tensor):
                    pred_shape = predicted_coords.shape
                    target_shape = batch['target_seq'].to(self.device).shape
                    
                    if pred_shape != target_shape:
                        print(f"WARNING: Shape mismatch: predictions {pred_shape} vs targets {target_shape}")
                        
                        # Fix target shape if needed - handle sequences of different lengths
                        min_seq_len = min(pred_shape[1], target_shape[1])
                        
                        # Truncate to match
                        predicted_coords = predicted_coords[:, :min_seq_len, :]
                        targets = batch['target_seq'].to(self.device)[:, :min_seq_len, :]
                    else:
                        targets = batch['target_seq'].to(self.device)
                else:
                    targets = batch['target_seq'].to(self.device)
                
                # Calculate loss
                loss, loss_components = self.compute_loss(predicted_coords, targets)
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                # Track statistics
                total_loss += loss.item()
                for key in loss_components:
                    total_loss_components[key] += loss_components[key]
                n_batches += 1
                
                # Update progress bar
                pbar.set_description(f"Train Loss: {loss.item():.4f} (MSE: {loss_components['mse_loss']:.4f}, Direction: {loss_components['direction_loss']:.4f})")
                
                # Log to TensorBoard (every N batches)
                if self.tensorboard_writer and batch_idx % 10 == 0:
                    global_step = epoch * len(self.train_loader) + batch_idx
                    self.tensorboard_writer.add_scalar('train/batch_loss', loss.item(), global_step)
                    for key, value in loss_components.items():
                        self.tensorboard_writer.add_scalar(f'train/batch_{key}', value, global_step)
                
            except Exception as e:
                error_count += 1
                # Print more details about the error
                import traceback
                print(f"Error processing batch: {e}")
                print(traceback.format_exc())
                
                if error_count > 5:  # Limit number of errors to display
                    print("Too many errors, stopping epoch early")
                    break
        
        if n_batches == 0:
            return float('inf'), {k: float('inf') for k in total_loss_components}
        
        # Calculate average losses
        avg_loss = total_loss / n_batches
        avg_loss_components = {k: v / n_batches for k, v in total_loss_components.items()}
        
        # Log epoch-level metrics to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('train/epoch_loss', avg_loss, epoch)
            for key, value in avg_loss_components.items():
                self.tensorboard_writer.add_scalar(f'train/epoch_{key}', value, epoch)
            
        return avg_loss, avg_loss_components


    def validate(self, epoch):
        """
        Validate the model on the validation set with tensor shape checking
        
        Args:
            epoch: Current epoch number (for TensorBoard logging)
            
        Returns:
            avg_loss: Average loss for the epoch
            avg_loss_components: Average loss components for the epoch
        """
        if self.val_loader is None:
            return None, None
            
        self.model.eval()
        total_loss = 0
        total_loss_components = {'mse_loss': 0, 'direction_loss': 0}
        n_batches = 0
        error_count = 0
        
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validating"):
                try:
                    # Create padded batch
                    batch = create_padded_batch(batch_data)
                    
                    # Forward pass with graph encoding disabled if specified
                    predicted_coords = self.model(batch)
                    
                    # Check shapes before loss computation
                    if isinstance(predicted_coords, torch.Tensor) and isinstance(batch['target_seq'], torch.Tensor):
                        pred_shape = predicted_coords.shape
                        target_shape = batch['target_seq'].to(self.device).shape
                        
                        if pred_shape != target_shape:
                            # Fix target shape if needed
                            min_seq_len = min(pred_shape[1], target_shape[1])
                            predicted_coords = predicted_coords[:, :min_seq_len, :]
                            targets = batch['target_seq'].to(self.device)[:, :min_seq_len, :]
                        else:
                            targets = batch['target_seq'].to(self.device)
                    else:
                        targets = batch['target_seq'].to(self.device)
                    
                    # Calculate loss
                    loss, loss_components = self.compute_loss(predicted_coords, targets)
                    
                    # Track statistics
                    total_loss += loss.item()
                    for key in loss_components:
                        total_loss_components[key] += loss_components[key]
                    n_batches += 1
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 3:  # Limit error messages
                        import traceback
                        print(f"Error during validation: {e}")
                        print(traceback.format_exc())
        
        if n_batches == 0:
            print("No valid batches processed during validation")
            return float('inf'), {k: float('inf') for k in total_loss_components}
            
        # Calculate average losses
        avg_loss = total_loss / n_batches
        avg_loss_components = {k: v / n_batches for k, v in total_loss_components.items()}
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('val/epoch_loss', avg_loss, epoch)
            for key, value in avg_loss_components.items():
                self.tensorboard_writer.add_scalar(f'val/epoch_{key}', value, epoch)
            
        return avg_loss, avg_loss_components
        
    def train(self, num_epochs, checkpoint_dir=None, initial_batch_size=None, 
            max_batch_size=None, batch_increase_epochs=5, evaluator=None, test_dataset=None):
        """
        Train the model for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train for
            checkpoint_dir: Directory to save checkpoints (optional)
            initial_batch_size: Starting batch size (optional)
            max_batch_size: Maximum batch size to try (optional)
            batch_increase_epochs: Number of epochs before trying to increase batch size
            evaluator: RouteEvaluator instance for visualization (optional)
            test_dataset: Test dataset for visualization if evaluator is not provided (optional)
            
        Returns:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
        """
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        # Track batch size adaptation
        current_batch_size = initial_batch_size
        batch_size_attempts = 0
        
        # Create evaluator if not provided but test_dataset is available
        if evaluator is None and test_dataset is not None:
            evaluator = RouteEvaluator(self.model, test_dataset, device=self.device, disable_graph=self.disable_graph)
            
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            train_loss, train_loss_components = self.train_epoch(epoch)
            train_losses.append(train_loss)
            
            # Validate
            val_loss, val_loss_components = self.validate(epoch)
            if val_loss is not None:
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} (MSE: {train_loss_components['mse_loss']:.4f}, Direction: {train_loss_components['direction_loss']:.4f}), "
                      f"Val Loss: {val_loss:.4f} (MSE: {val_loss_components['mse_loss']:.4f}, Direction: {val_loss_components['direction_loss']:.4f})")
                
                # Update learning rate scheduler
                self.scheduler.step(val_loss)
                
                # Log learning rate to TensorBoard
                if self.tensorboard_writer:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.tensorboard_writer.add_scalar('train/learning_rate', current_lr, epoch)
                
                # Save checkpoint if this is the best model so far
                if checkpoint_dir and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'disable_graph': self.disable_graph,
                    }, os.path.join(checkpoint_dir, f'best_model.pt'))
                    print(f"✓ Saved best model with validation loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} (MSE: {train_loss_components['mse_loss']:.4f}, Direction: {train_loss_components['direction_loss']:.4f})")
            
            # Try increasing batch size if conditions are met
            if (initial_batch_size is not None and max_batch_size is not None and
                current_batch_size < max_batch_size and 
                (epoch + 1) % batch_increase_epochs == 0):
                
                # Only attempt if previous training was successful
                if train_loss < float('inf') and (val_loss is None or val_loss < float('inf')):
                    batch_size_attempts += 1
                    new_batch_size = min(current_batch_size * 2, max_batch_size)
                    
                    print(f"\nIncreasing batch size from {current_batch_size} to {new_batch_size}")
                    
                    # Create new data loaders with increased batch size
                    self.train_loader.batch_size = new_batch_size
                    if self.val_loader:
                        self.val_loader.batch_size = new_batch_size
                    
                    current_batch_size = new_batch_size
                    
                    # Log batch size change to TensorBoard
                    if self.tensorboard_writer:
                        self.tensorboard_writer.add_scalar('train/batch_size', current_batch_size, epoch)
            
            # Save checkpoint every N epochs
            if checkpoint_dir and (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss if val_loss is not None else None,
                    'disable_graph': self.disable_graph,
                }, os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt'))

            # Visualize 4 route comparisons using the evaluator
            if checkpoint_dir and evaluator:
                print(f"Visualizing 4 route examples for epoch {epoch+1}...")
                
                # Make sure the model in the evaluator is updated with the current model weights
                evaluator.model.load_state_dict(self.model.state_dict())
                
                # Log model graph to TensorBoard (only once)
                if self.tensorboard_writer and epoch == 0:
                    try:
                        # Try to log model architecture if possible
                        # This might fail depending on the model structure
                        dummy_input = torch.zeros(1, 10, 2).to(self.device)  # Adjust shape as needed
                        self.tensorboard_writer.add_graph(self.model, dummy_input)
                    except:
                        print("Could not log model graph to TensorBoard")
                
                # Visualize 4 examples
                for i in range(4):
                    viz_path = os.path.join(checkpoint_dir, f'route_comparison_epoch{epoch}_example{i}.png')
                    try:
                        # Visualize route comparison using the evaluator
                        fig = evaluator.visualize_route_comparison(i, save_path=viz_path)
                        
                        # Log figure to TensorBoard
                        if self.tensorboard_writer:
                            self.tensorboard_writer.add_figure(
                                f'evaluation/route_example_{i}', 
                                fig,
                                global_step=epoch
                            )
                        
                        plt.close(fig)  # Close the figure to free memory
                    except Exception as e:
                        print(f"Error visualizing example {i}: {e}")
                
                # Run evaluation metrics (if not too expensive)
                try:
                    eval_metrics = evaluator.evaluate_generated_routes(num_samples=min(20, len(evaluator.test_dataset)))
                    print(f"Evaluation metrics - MSE: {eval_metrics['mse']:.4f}, Distance Error: {eval_metrics['distance_error']:.2f}, End Point Error: {eval_metrics['end_point_error']:.4f}")
                    
                    # Log eval metrics to TensorBoard
                    if self.tensorboard_writer:
                        for key, value in eval_metrics.items():
                            self.tensorboard_writer.add_scalar(f'evaluation/{key}', value, epoch)
                except Exception as e:
                    print(f"Error during evaluation: {e}")
            
            # For backward compatibility, use model's visualize_route_comparison if it exists and no evaluator provided
            elif checkpoint_dir and hasattr(self.model, 'visualize_route_comparison'):
                viz_path = os.path.join(checkpoint_dir, f'route_comparison_{epoch}.png')
                # visualize route comparison
                fig = self.model.visualize_route_comparison(0, save_path=viz_path)
                
                # Log figure to TensorBoard
                if self.tensorboard_writer:
                    self.tensorboard_writer.add_figure('evaluation/route_example', fig, global_step=epoch)
        
        # Close TensorBoard writer
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        return train_losses, val_losses


# Modified RouteEvaluator to support TensorBoard logging
class RouteEvaluator:
    """Class to evaluate the quality of generated routes"""
    def __init__(self, model, test_dataset, device='mps', disable_graph=False, tensorboard_writer=None):
        self.model = model
        self.test_dataset = test_dataset
        self.device = device
        self.disable_graph = disable_graph
        self.tensorboard_writer = tensorboard_writer
        self.model.to(device)
        
        print(f"Evaluator initialized with:")
        print(f"  - Device: {device}")
        print(f"  - Graph encoding: {'Disabled' if disable_graph else 'Enabled'}")
        print(f"  - Test dataset size: {len(test_dataset)}")
        print(f"  - TensorBoard: {'Enabled' if tensorboard_writer else 'Disabled'}")
    
    def evaluate_generated_routes(self, num_samples=10):
        """
        Evaluate the model by generating routes and comparing to ground truth
        
        Args:
            num_samples: Number of samples to evaluate
            
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        self.model.eval()
        metrics = {
            'mse': [],
            'distance_error': [],
            'end_point_error': []
        }
        
        route_details = []  # Store details for potential logging
        successful_samples = 0
        errors = 0
        
        with torch.no_grad():
            # Create progress bar
            pbar = tqdm(range(min(num_samples, len(self.test_dataset))), desc="Evaluating")
            
            for i in pbar:
                try:
                    sample = self.test_dataset[i]
                    
                    # Extract route information
                    route_xy = sample['route'].numpy()
                    start_lat, start_lon = route_xy[0]
                    end_lat, end_lon = route_xy[-1]
                    target_distance = sample['conditions'][0].item()
                    
                    # Generate route with graph encoding enabled/disabled as configured
                    generated_route = self.model.generate_route(
                        sample['graph'],
                        start_lat, start_lon,
                        end_lat, end_lon,
                        target_distance,
                        sample['nearest_nodes'],
                        sample['subgraph'],
                        device=self.device,
                    ).cpu().numpy()
                    
                    # Calculate metrics
                    mse = np.mean((generated_route - route_xy)**2)
                    
                    # Calculate actual distances
                    actual_distance = get_route_distance(list(route_xy[:,0]), list(route_xy[:,1]))
                    generated_distance = get_route_distance(list(generated_route[:,0]), list(generated_route[:,1]))
                    
                    if actual_distance > 0:
                        distance_error = abs(generated_distance - actual_distance) / actual_distance
                    else:
                        distance_error = float('inf')
                    
                    # End point error
                    end_point_error = np.sqrt(
                        (generated_route[-1, 0] - end_lat)**2 + 
                        (generated_route[-1, 1] - end_lon)**2
                    )
                    
                    # Add to metrics
                    metrics['mse'].append(mse)
                    metrics['distance_error'].append(distance_error)
                    metrics['end_point_error'].append(end_point_error)
                    
                    # Store route details for potential logging
                    route_details.append({
                        'sample_idx': i,
                        'mse': mse,
                        'distance_error': distance_error,
                        'end_point_error': end_point_error,
                        'target_distance': target_distance,
                        'actual_distance': actual_distance,
                        'generated_distance': generated_distance
                    })
                    
                    # Update progress bar description
                    pbar.set_description(f"MSE: {mse:.4f}, Dist Err: {distance_error:.2f}")
                    
                    successful_samples += 1
                    
                except Exception as e:
                    errors += 1
                    if errors <= 3:  # Limit error messages
                        print(f"Error evaluating sample {i}: {e}")
        
        print(f"Evaluated {successful_samples} samples successfully, encountered {errors} errors")
                
        # Calculate averages
        if successful_samples > 0:
            for key in metrics:
                metrics[key] = np.mean(metrics[key])
            
            # Log histogram of metrics to TensorBoard if available
            if self.tensorboard_writer:
                # Extract individual metrics for histograms
                mse_values = [detail['mse'] for detail in route_details]
                distance_error_values = [detail['distance_error'] for detail in route_details]
                end_point_error_values = [detail['end_point_error'] for detail in route_details]
                
                # Log histograms
                self.tensorboard_writer.add_histogram('evaluation/mse_histogram', np.array(mse_values))
                self.tensorboard_writer.add_histogram('evaluation/distance_error_histogram', np.array(distance_error_values))
                self.tensorboard_writer.add_histogram('evaluation/end_point_error_histogram', np.array(end_point_error_values))
                
                # Create scatter plot of target vs. generated distance
                try:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 8))
                    target_distances = [detail['target_distance'] for detail in route_details]
                    generated_distances = [detail['generated_distance'] for detail in route_details]
                    ax.scatter(target_distances, generated_distances, alpha=0.7)
                    ax.plot([0, max(target_distances)], [0, max(target_distances)], 'r--')  # Ideal line
                    ax.set_xlabel('Target Distance (km)')
                    ax.set_ylabel('Generated Distance (km)')
                    ax.set_title('Target vs. Generated Route Distances')
                    ax.grid(True)
                    
                    # Add to TensorBoard
                    self.tensorboard_writer.add_figure('evaluation/target_vs_generated_distance', fig)
                    plt.close(fig)
                except Exception as e:
                    print(f"Error creating distance scatter plot: {e}")
        else:
            # Return NaN if no samples were processed successfully
            for key in metrics:
                metrics[key] = float('nan')
            
        return metrics
    
    def visualize_route_comparison(self, idx, save_path=None):
        """
        Visualize a comparison between generated and ground truth routes
        
        Args:
            idx: Index of the route to compare
            save_path: Path to save the visualization (optional)
            
        Returns:
            fig: matplotlib figure
        """
        
        sample = self.test_dataset[idx]
        route_xy = sample['route'].numpy()
        start_lat, start_lon = route_xy[0]
        end_lat, end_lon = route_xy[-1]
        target_distance = sample['conditions'][0].item()
        
        # Calculate actual distance
        actual_distance = get_route_distance(list(route_xy[:,0]), list(route_xy[:,1]))
        
        # Generate route with graph encoding enabled/disabled as configured
        with torch.no_grad():
            generated_route = self.model.generate_route(
                sample['graph'],
                start_lat, start_lon,
                end_lat, end_lon,
                target_distance,
                sample['nearest_nodes'],
                sample['subgraph'],
                device=self.device,
            ).cpu().numpy()
        
        # Calculate generated distance
        generated_distance = get_route_distance(list(generated_route[:,0]), list(generated_route[:,1]))
        
        # Calculate distance error
        if actual_distance > 0:
            distance_error = abs(generated_distance - actual_distance) / actual_distance * 100  # as percentage
        else:
            distance_error = float('inf')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        
        # Plot OSM basemap if osmnx is available and graphs are enabled
        if not self.disable_graph:
            try:
                bounds = sample['bounds']
                north, south, east, west = bounds[1], bounds[0], bounds[3], bounds[2]
                ox.plot_graph(sample['graph'], ax=ax, show=False, close=False, 
                              edge_color='gray', edge_alpha=0.2, node_size=0)
            except Exception as e:
                print(f"Warning: Could not plot graph background: {e}")

        # Plot the routes
        ax.plot(route_xy[:, 1], route_xy[:, 0], 'b-', linewidth=2, label='Ground Truth')
        ax.plot(generated_route[:, 1], generated_route[:, 0], 'r-', linewidth=2, label='Generated')
        
        # Plot start and end points
        ax.plot(start_lon, start_lat, 'go', markersize=10, label='Start')
        ax.plot(end_lon, end_lat, 'mo', markersize=10, label='End')
        
        # Add legend and title
        ax.legend()
        title = f'Route Comparison\n'
        title += f'Target: {target_distance:.2f} km, Actual: {actual_distance:.2f} km, Generated: {generated_distance:.2f} km\n'
        title += f'Distance Error: {distance_error:.1f}%'
        if self.disable_graph:
            title += ' (Graph Disabled)'
            
        ax.set_title(title)
        
        # Save figure if needed
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig


# Define a picklable collate function (must be at module level for pickling)
def identity_collate(x):
    return x

def main():

    default_device = (
    'mps' if torch.backends.mps.is_available() 
    else 'cuda' if torch.cuda.is_available() 
    else 'cpu'
    )
    # Set up command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train running route prediction model')
    parser.add_argument('--data_path', type=str, default="data/processed_combined.csv", help='Path to dataset CSV')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=8, help='Initial batch size for training')
    parser.add_argument('--max_batch_size', type=int, default=32, help='Maximum batch size to try during training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--n_hops', type=int, default=2, help='Number of hops in the graph')
    parser.add_argument('--disable_graph', action='store_true', help='Disable graph encoder and use only coordinate data')
    parser.add_argument('--device', type=str, default=default_device, 
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint for evaluation')
    
    # Add new arguments for parallel graph loading
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for parallel graph loading')
    parser.add_argument('--max_workers', type=int, help='Maximum workers for parallel graph loading')
    parser.add_argument('--cache_dir', type=str, default='test_cache', help='Directory to cache graph data')
    parser.add_argument('--disable_parallel', action='store_true', help='Disable parallel graph loading')
    parser.add_argument('--proximity_threshold', type=float, default=0.001, 
                        help='Threshold for considering bounds as similar (in degrees)')
    parser.add_argument('--process_batch_size', type=int, default=10, 
                        help='Batch size for parallel processing')
    parser.add_argument('--clear_cache', action='store_true', 
                        help='Clear the graph cache before loading')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to load for training')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Clear cache if requested
    if args.clear_cache and args.cache_dir and os.path.exists(args.cache_dir):
        import shutil
        print(f"Clearing cache directory: {args.cache_dir}")
        shutil.rmtree(args.cache_dir)
        os.makedirs(args.cache_dir, exist_ok=True)
    
    # Load dataset with error handling
    try:
        print(f"Loading dataset from {args.data_path}...")
        dataset = RunningRouteDataset(
            args.data_path, 
            verbose=True, 
            n_hops=args.n_hops,
            cache_dir=args.cache_dir,
            num_workers=args.num_workers,
            max_workers=args.max_workers,
            batch_size=args.process_batch_size,
            parallel_loading=not args.disable_parallel,
            proximity_threshold=args.proximity_threshold,
            num_samples = args.num_samples
        )
        print(f"Successfully loaded {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    
    # Split into train, val, test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"Dataset splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create data loaders - use single process (num_workers=0) to avoid pickling issues
    # with complex graph objects 
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use single process to avoid pickling issues
        collate_fn=identity_collate
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Use single process to avoid pickling issues
        collate_fn=identity_collate
    )
    
    # Initialize model
    model = GraphAwareRouteGenerator(
        graph_feature_dim=256,
        condition_dim=64,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        n_hops=args.n_hops
    )
    
    # Load model if evaluation only
    if args.eval_only and args.model_path:
        try:
            print(f"Loading model from {args.model_path}...")
            checkpoint = torch.load(args.model_path, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Check if model was trained with graph disabled
            disable_graph = checkpoint.get('disable_graph', False)
            if disable_graph:
                print("Note: This model was trained with graph encoding disabled")
            
            print(f"Successfully loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            
            # Run evaluation
            evaluator = RouteEvaluator(model, test_dataset, device=args.device, 
                                       disable_graph=disable_graph)
            metrics = evaluator.evaluate_generated_routes(num_samples=20)
            
            print("\nEvaluation Results:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # Create visualization directory if needed
            viz_dir = os.path.join(args.checkpoint_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Visualize some examples
            print("\nGenerating route visualizations...")
            for i in range(5):
                idx = np.random.randint(len(test_dataset))
                viz_path = os.path.join(viz_dir, f'route_comparison_{i}.png')
                
                try:
                    fig = evaluator.visualize_route_comparison(idx, save_path=viz_path)
                    plt.close(fig)
                    print(f"  Generated visualization {i+1}/5: {viz_path}")
                except Exception as e:
                    print(f"  Error generating visualization {i+1}: {e}")
            
        except Exception as e:
            print(f"Error loading or evaluating model: {e}")
            import traceback
            traceback.print_exc()
        
        return
    
    # Initialize trainer with graph settings
    trainer = RouteModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.learning_rate,
        device=args.device,
        disable_graph=args.disable_graph,
        verbose=True,
        tensorboard_log_dir=f'tensorboard_logs/{get_datetime_string()}'
    )
    
    # Print adaptive batch size settings if enabled
    if args.max_batch_size > args.batch_size:
        print(f"Starting with batch size {args.batch_size}, will try to increase up to {args.max_batch_size}")
    
    # Train the model with batch size adaptation
    train_losses, val_losses = trainer.train(
        num_epochs=args.num_epochs,
        checkpoint_dir=args.checkpoint_dir,
        initial_batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        batch_increase_epochs=5  # Try increasing batch size every 5 epochs
    )
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.checkpoint_dir, 'loss_curve.png'))
    
    # Load the best model for evaluation
    best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.4f}")
    
    # Run evaluation
    evaluator = RouteEvaluator(model, test_dataset, device=args.device)
    metrics = evaluator.evaluate_generated_routes(num_samples=20)
    
    print("Evaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Visualize some examples
    for i in range(5):
        idx = np.random.randint(len(test_dataset))
        fig = evaluator.visualize_route_comparison(
            idx, save_path=os.path.join(args.checkpoint_dir, f'route_comparison_{i}.png')
        )
        plt.close(fig)


import cProfile
import pstats
import time
import os
import torch
import numpy as np
from torch_geometric.data import Batch

def profiler_main():
    """Profile the performance of the route prediction model for one training step"""
    # Set up minimal arguments for profiling
    class Args:
        data_path = "data/processed_for_graph.csv"  # Replace with your actual path
        batch_size = 4  # Small batch size for profiling
        hidden_dim = 256
        n_hops = 2
        device = 'mps'
        seed = 42
        cache_dir = 'test_cache'
        num_workers = 4
        max_workers = None
        process_batch_size = 10
        proximity_threshold = 0.001
        disable_parallel = False
        disable_graph = False
        clear_cache = False
        num_samples = 100
        
    args = Args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=== Starting Performance Profiling ===")
    
    # Profile dataset loading
    start_time = time.time()
    print(f"Loading dataset from {args.data_path}...")
    try:
        dataset = RunningRouteDataset(
            args.data_path, 
            verbose=True, 
            n_hops=args.n_hops,
            cache_dir=args.cache_dir,
            num_workers=args.num_workers,
            max_workers=args.max_workers,
            batch_size=args.process_batch_size,
            parallel_loading=not args.disable_parallel,
            proximity_threshold=args.proximity_threshold,
            num_samples=args.num_samples
        )
        dataset_time = time.time() - start_time
        print(f"Dataset loading took {dataset_time:.2f} seconds for {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create a small subset for profiling
    subset_size = min(args.batch_size * 3, len(dataset))
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    profile_dataset = torch.utils.data.Subset(dataset, indices)
    
    # Create data loader
    start_time = time.time()
    train_loader = torch.utils.data.DataLoader(
        profile_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use single process for profiling
        collate_fn=identity_collate
    )
    loader_time = time.time() - start_time
    print(f"DataLoader creation took {loader_time:.2f} seconds")
    
    # Initialize model
    start_time = time.time()
    model = GraphAwareRouteGenerator(
        graph_feature_dim=256,
        condition_dim=64,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        n_hops=args.n_hops
    )
    model = model.to(args.device)
    model_init_time = time.time() - start_time
    print(f"Model initialization took {model_init_time:.2f} seconds")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Profile one batch processing
    print("\n=== Profiling Single Batch Processing ===")
    
    # Get a single batch
    try:
        batch_data = next(iter(train_loader))
        
        # Profile batch preparation
        start_time = time.time()
        batch_dict = create_padded_batch(batch_data)
        prepare_batch_time = time.time() - start_time
        print(f"Batch preparation took {prepare_batch_time:.2f} seconds")
        
        # Profile graph batch creation
        start_time = time.time()
        device = args.device
        input_lat = batch_dict['input_lat']
        input_lon = batch_dict['input_lon']
        subgraphs = batch_dict['subgraph']
        # graph_batch = model.prepare_graph_batch(graphs, input_lat, input_lon, device)
        graph_batch = subgraphs
        graph_batch_time = time.time() - start_time
        print(f"Graph batch creation took {graph_batch_time:.2f} seconds")
        
        # Profile graph encoding
        start_time = time.time()
        graph_features = model.graph_encoder(graph_batch)
        graph_encoding_time = time.time() - start_time
        print(f"Graph encoding took {graph_encoding_time:.2f} seconds")
        
        # Profile entire forward pass
        start_time = time.time()
        batch_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
        outputs = model(batch_dict)
        forward_time = time.time() - start_time
        print(f"Forward pass took {forward_time:.2f} seconds")
        
        # Profile loss computation
        start_time = time.time()
        loss = criterion(outputs, batch_dict['target_seq'])
        loss_time = time.time() - start_time
        print(f"Loss computation took {loss_time:.2f} seconds")
        
        # Profile backward pass
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        print(f"Backward pass took {backward_time:.2f} seconds")
        
        # Profile optimizer step
        start_time = time.time()
        optimizer.step()
        optimizer.zero_grad()
        optim_time = time.time() - start_time
        print(f"Optimizer step took {optim_time:.2f} seconds")
        
        # Summary of timings
        print("\n=== Profiling Summary ===")
        print(f"Dataset loading: {dataset_time:.2f} s")
        print(f"DataLoader creation: {loader_time:.2f} s")
        print(f"Model initialization: {model_init_time:.2f} s")
        print(f"Batch preparation: {prepare_batch_time:.2f} s")
        print(f"Graph batch creation: {graph_batch_time:.2f} s")
        print(f"Graph encoding: {graph_encoding_time:.2f} s")
        print(f"Full forward pass: {forward_time:.2f} s")
        print(f"Loss computation: {loss_time:.2f} s")
        print(f"Backward pass: {backward_time:.2f} s")
        print(f"Optimizer step: {optim_time:.2f} s")
        
        # Total time for one training step
        total_step_time = prepare_batch_time + forward_time + loss_time + backward_time + optim_time
        print(f"\nTotal time for one training step: {total_step_time:.2f} s")
        
        # Show batch details
        print("\n=== Batch Statistics ===")
        print(f"Batch size: {args.batch_size}")
        if hasattr(graph_batch, 'x'):
            print(f"Total nodes in batch: {graph_batch.x.shape[0]}")
        if hasattr(graph_batch, 'edge_index'):
            print(f"Total edges in batch: {graph_batch.edge_index.shape[1]}")
        
        # Estimate time for one epoch
        batch_count = len(dataset) // args.batch_size
        estimated_epoch_time = total_step_time * batch_count
        print(f"\nEstimated time for one epoch ({batch_count} batches): {estimated_epoch_time:.2f} s ({estimated_epoch_time/60:.2f} min)")
        
    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
