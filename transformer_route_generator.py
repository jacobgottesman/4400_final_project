import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# Import the model architecture
from transformer import RouteTransformer

def load_model(model_path, device):
    """
    Load a trained transformer model
    """
    print(f"Loading model from {model_path}")
    model = RouteTransformer().to(device)
    
    # Load the state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set to evaluation mode
    model.eval()
    
    return model

def generate_route(model, map_path, conditions, device, normalize_stats=None, temperature=0.5, num_routes=1):
    """
    Generate a route on a map with specified conditions
    
    Args:
        model: Trained transformer model
        map_path: Path to the map image
        conditions: List of [distance, duration, start_end_dist, heart_rate]
        device: Torch device
        normalize_stats: Optional normalization statistics for conditions
        temperature: Temperature parameter for generation (higher = more diverse)
        num_routes: Number of routes to generate
        
    Returns:
        Generated routes as numpy array, Map as numpy array
    """
    # Load and preprocess the map
    transform = transforms.Compose([
        transforms.Resize((683, 1366)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    map_img = Image.open(map_path).convert('RGB')
    map_tensor = transform(map_img).unsqueeze(0).to(device)
    
    # Get image dimensions for later denormalization
    img_width, img_height = map_img.size
    
    # Normalize conditions if normalization stats are provided
    if normalize_stats:
        normalized_conditions = []
        condition_names = ['distance', 'duration', 'start_end_dist', 'heart_rate']
        
        for i, (name, value) in enumerate(zip(condition_names, conditions)):
            if name in normalize_stats:
                min_val = normalize_stats[name]['min']
                max_val = normalize_stats[name]['max']
                if max_val > min_val:
                    normalized_val = (value - min_val) / (max_val - min_val)
                else:
                    normalized_val = value
                normalized_conditions.append(normalized_val)
            else:
                normalized_conditions.append(value)
        
        conditions = normalized_conditions
    
    # Convert conditions to tensor
    conditions_tensor = torch.tensor(conditions, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Generate multiple routes if requested
    all_routes = []
    for i in range(num_routes):
        # Generate route with transformer
        with torch.no_grad():
            generated_route = model.generate_route(
                map_tensor, 
                conditions_tensor, 
                temperature=temperature
            )
        
        # Convert to numpy for visualization
        route_np = generated_route.squeeze(0).cpu().numpy()
        
        # Denormalize route coordinates to pixel values
        route_pixels = route_np.copy()
        route_pixels[:, 0] *= img_width
        route_pixels[:, 1] *= img_height
        
        all_routes.append(route_pixels)
    
    # Convert map tensor to numpy for visualization
    map_np = map_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    map_np = (map_np * 0.5) + 0.5  # Unnormalize
    
    return all_routes, map_np

def visualize_route(routes, map_img, save_path=None, show_plot=True):
    """
    Visualize the generated route(s) on the map
    """
    plt.figure(figsize=(15, 10))
    
    # Display the map
    plt.imshow(map_img)
    
    # Define a colormap for multiple routes
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    # Overlay each route with a different color
    for i, route in enumerate(routes):
        color = colors[i % len(colors)]
        plt.plot(route[:, 0], route[:, 1], f'{color}-', linewidth=2, label=f'Route {i+1}')
        
        # Highlight start and end points
        plt.plot(route[0, 0], route[0, 1], f'{color}o', markersize=8)
        plt.plot(route[-1, 0], route[-1, 1], f'{color}o', markersize=8)
    
    if len(routes) > 1:
        plt.legend()
        
    plt.title("Generated Running Routes")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def save_route_to_csv(route, output_path):
    """
    Save the generated route to a CSV file
    """
    # Save to CSV
    with open(output_path, 'w') as f:
        f.write("point_id,x,y\n")
        for i, (x, y) in enumerate(route):
            f.write(f"{i},{x:.2f},{y:.2f}\n")
    
    print(f"Saved route coordinates to {output_path}")

def calculate_route_statistics(route):
    """
    Calculate basic statistics about the route
    """
    # Calculate total path length
    total_length = 0
    for i in range(len(route) - 1):
        dx = route[i+1, 0] - route[i, 0]
        dy = route[i+1, 1] - route[i, 1]
        segment_length = np.sqrt(dx**2 + dy**2)
        total_length += segment_length
    
    # Calculate start-end distance
    start_end_dist = np.sqrt(
        (route[-1, 0] - route[0, 0])**2 + 
        (route[-1, 1] - route[0, 1])**2
    )
    
    # Calculate route complexity (approximated by number of turns)
    turns = 0
    for i in range(1, len(route) - 1):
        # Calculate vectors
        v1 = route[i] - route[i-1]
        v2 = route[i+1] - route[i]
        
        # Calculate angle
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # Avoid division by zero
        if norm_v1 > 0 and norm_v2 > 0:
            cos_angle = dot_product / (norm_v1 * norm_v2)
            # Clamp to avoid numerical errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # Count as turn if angle is significant (> 30 degrees)
            if np.degrees(angle) > 30:
                turns += 1
    
    return {
        'total_length': total_length,
        'start_end_distance': start_end_dist,
        'turns': turns,
        'points': len(route)
    }

def main():
    parser = argparse.ArgumentParser(description="Generate running routes using a trained Transformer model")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained transformer model")
    parser.add_argument("--map", type=str, required=True, help="Path to the map image")
    parser.add_argument("--output", type=str, default="generated_route", help="Output prefix for saved files")
    parser.add_argument("--distance", type=float, default=5.0, help="Route distance (km)")
    parser.add_argument("--duration", type=float, default=30.0, help="Route duration (minutes)")
    parser.add_argument("--start_end_dist", type=float, default=0.1, help="Start-end distance")
    parser.add_argument("--heart_rate", type=float, default=150.0, help="Average heart rate")
    parser.add_argument("--norm_stats", type=str, help="Path to normalization statistics JSON file")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for generation (higher = more diverse)")
    parser.add_argument("--num_routes", type=int, default=1, help="Number of routes to generate")
    parser.add_argument("--no_display", action="store_true", help="Don't display the generated routes (only save)")
    
    args = parser.parse_args()
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load the transformer model
    model = load_model(args.model, device)
    print("Model loaded successfully")
    
    # Load normalization statistics if provided
    norm_stats = None
    if args.norm_stats and os.path.exists(args.norm_stats):
        with open(args.norm_stats, 'r') as f:
            norm_stats = json.load(f)
        print("Loaded normalization statistics")
    
    # Prepare conditions
    conditions = [args.distance, args.duration, args.start_end_dist, args.heart_rate]
    print(f"Generating route with conditions: {conditions}")
    
    # Generate route(s)
    routes, map_np = generate_route(
        model, 
        args.map, 
        conditions, 
        device,
        normalize_stats=norm_stats,
        temperature=args.temperature,
        num_routes=args.num_routes
    )
    print(f"Generated {len(routes)} route(s)")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Visualize and save the route(s)
    visualize_route(
        routes, 
        map_np, 
        save_path=f"{args.output}.png", 
        show_plot=not args.no_display
    )
    
    # Save routes and calculate statistics
    for i, route in enumerate(routes):
        if args.num_routes > 1:
            output_prefix = f"{args.output}_{i+1}"
        else:
            output_prefix = args.output
        
        # Save route coordinates to CSV
        save_route_to_csv(route, f"{output_prefix}.csv")
        
        # Calculate and save route statistics
        stats = calculate_route_statistics(route)
        with open(f"{output_prefix}_stats.json", 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"Route {i+1} statistics:")
        print(f"- Total length: {stats['total_length']:.2f} pixels")
        print(f"- Start-end distance: {stats['start_end_distance']:.2f} pixels")
        print(f"- Number of turns: {stats['turns']}")
        print(f"- Number of points: {stats['points']}")

if __name__ == "__main__":
    main()