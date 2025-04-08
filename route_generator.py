import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

# Import the model architecture from the main script
# Assuming you've saved the model definition in route_cgan_model.py
from cgan import Generator


def load_model(model_path, device):
    """
    Load a trained generator model
    """
    generator = Generator().to(device)
    
    # Load the state dict
    generator.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set to evaluation mode
    generator.eval()
    
    return generator


def generate_route(generator, map_path, conditions, device, normalize_stats=None):
    """
    Generate a route on a map with specified conditions
    
    Args:
        generator: Trained generator model
        map_path: Path to the map image
        conditions: List of [distance, duration, start_end_dist, heart_rate]
        device: Torch device
        normalize_stats: Optional normalization statistics for conditions
    
    Returns:
        Generated route as numpy array, Map as numpy array
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
    
    # Generate random noise
    noise = torch.randn(1, 128).to(device)
    
    # Generate route
    with torch.no_grad():
        generated_route = generator(map_tensor, conditions_tensor, noise)
    
    # Convert to numpy for visualization
    route_np = generated_route.squeeze(0).cpu().numpy()
    map_np = map_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    map_np = (map_np * 0.5) + 0.5  # Unnormalize
    
    # Denormalize route coordinates to pixel values
    route_pixels = route_np.copy()
    route_pixels[:, 0] *= img_width
    route_pixels[:, 1] *= img_height
    
    return route_pixels, map_np


def visualize_route(route, map_img, save_path=None):
    """
    Visualize the generated route on the map
    """
    plt.figure(figsize=(15, 10))
    
    # Display the map
    plt.imshow(map_img)
    
    # Overlay the route (scale coordinates to map dimensions)
    plt.plot(route[:, 0] * map_img.shape[1], route[:, 1] * map_img.shape[0], 'r-', linewidth=2)
    
    plt.title("Generated Running Route")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def save_route_to_csv(route, map_dimensions, output_path):
    """
    Save the generated route to a CSV file
    Converts the normalized [0,1] coordinates back to image pixel coordinates
    """
    # Scale route to image dimensions
    width, height = map_dimensions
    route_pixels = route.copy()
    route_pixels[:, 0] *= width
    route_pixels[:, 1] *= height
    
    # Save to CSV
    with open(output_path, 'w') as f:
        f.write("point_id,x,y\n")
        for i, (x, y) in enumerate(route_pixels):
            f.write(f"{i},{x:.2f},{y:.2f}\n")
    
    print(f"Saved route coordinates to {output_path}")
    
    return route_pixels


def main():
    parser = argparse.ArgumentParser(description="Generate running routes using a trained CGAN model")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained generator model")
    parser.add_argument("--map", type=str, required=True, help="Path to the map image")
    parser.add_argument("--output", type=str, default="generated_route", help="Output prefix for saved files")
    parser.add_argument("--distance", type=float, default=5.0, help="Route distance (km)")
    parser.add_argument("--duration", type=float, default=30.0, help="Route duration (minutes)")
    parser.add_argument("--start_end_dist", type=float, default=0.1, help="Start-end distance")
    parser.add_argument("--heart_rate", type=float, default=150.0, help="Average heart rate")
    parser.add_argument("--norm_stats", type=str, help="Path to normalization statistics JSON file")
    
    args = parser.parse_args()
    
    # Set device
    # Check if GPU is available, else MPS, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load the generator model
    generator = load_model(args.model, device)
    print("Model loaded successfully")
    
    # Load normalization statistics if provided
    norm_stats = None
    if args.norm_stats and os.path.exists(args.norm_stats):
        with open(args.norm_stats, 'r') as f:
            norm_stats = json.load(f)
        print("Loaded normalization statistics")
    
    # Prepare conditions
    conditions = [args.distance, args.duration, args.start_end_dist, args.heart_rate]
    
    # Generate route
    route, map_np = generate_route(generator, args.map, conditions, device)
    print(f"Generated route with {len(route)} points")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Visualize and save the route
    visualize_route(route, map_np, f"{args.output}.png")
    
    # Save route coordinates to CSV
    save_route_to_csv(route, (map_np.shape[1], map_np.shape[0]), f"{args.output}.csv")


if __name__ == "__main__":
    main()