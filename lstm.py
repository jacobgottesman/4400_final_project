import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import datetime

class RunningRouteDataset(Dataset):
    def __init__(self, csv_path, img_dirs, transform=None, verbose=False):
        """
        Dataset for running routes and map images
        
        Args:
            csv_path: Path to CSV file with metadata and route information
            img_dirs: List of directories containing map images
            transform: Transforms to apply to images
            verbose: Whether to print verbose logs during data loading
        """
        print(f"Loading dataset from {csv_path}...")
        self.data = pd.read_csv(csv_path)
        # self.data = self.data.iloc[0:1000, :]
        self.data['route_xy'] = self.data['route_xy'].apply(eval)
        self.img_dirs = img_dirs
        self.transform = transform
        self.verbose = verbose
        
        print(f"Processing heart rate data for {len(self.data)} routes...")
        # Preprocess heart_rate data
        self.data['avg_heart_rate'] = self.data['heart_rate'].apply(
            lambda x: np.mean(eval(x)) if isinstance(x, str) else (
                np.mean(x) if isinstance(x, list) else x
            )
        )
        print("Dataset preparation complete!")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the row from the dataframe
        row = self.data.iloc[idx]
        
        # Load image
        img_id = row['index']
        
        if self.verbose:
            print(f"Loading image for index {idx}, Image ID: {img_id}")
        
        if img_id > 50425:
            img_path = os.path.join(self.img_dirs[1], f"map{img_id}.png")
        else:
            img_path = os.path.join(self.img_dirs[0], f"map{img_id}.png")

        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get route coordinates
        route_xy = np.array(eval(row['route_xy'])) if isinstance(row['route_xy'], str) else row['route_xy']
        route_tensor = torch.tensor(route_xy, dtype=torch.float32)
        
        # Get conditional parameters
        distance = torch.tensor([row['distance']], dtype=torch.float32)
        
        # Extract start and end points
        start_point = torch.tensor(route_xy[0], dtype=torch.float32)
        end_point = torch.tensor(route_xy[-1], dtype=torch.float32)
        
        # Combine conditions into one tensor
        conditions = torch.cat([distance, start_point, end_point], dim=0)
        
        # Create input-target pairs for sequence learning
        # Input: all coordinates except the last one
        # Target: all coordinates except the first one
        input_seq = route_tensor[:-1]
        target_seq = route_tensor[1:]
        
        return {
            'image': image,
            'route': route_tensor,
            'input_seq': input_seq,
            'target_seq': target_seq,
            'conditions': conditions,
            'seq_length': len(route_xy)
        }


class MapEncoder(nn.Module):
    """Encodes the map image into a feature representation"""
    def __init__(self, output_dim=256):
        super(MapEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Calculate the flattened size - for 683x1366 input
        # After 4 stride-2 layers: 43x86x128
        self.flattened_size = 128 * 43 * 86
        
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class ConditionEncoder(nn.Module):
    """Encodes the conditioning information (distance, start, end points)"""
    def __init__(self, input_dim=5, output_dim=64):
        super(ConditionEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


class RoutePredictor(nn.Module):
    """LSTM model for predicting the next coordinate based on previous coordinates"""
    def __init__(self, map_feature_dim=256, condition_dim=64, hidden_dim=256, num_layers=2):
        super(RoutePredictor, self).__init__()
        
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
        
        # Context integration layer - combines LSTM output with map and condition features
        self.context_layer = nn.Sequential(
            nn.Linear(hidden_dim + map_feature_dim + condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Final coordinate prediction layer
        self.output_layer = nn.Linear(64, 2)
    
    def forward(self, coords, map_features, condition_features, hidden=None):
        """
        Forward pass through the route predictor
        
        Args:
            coords: Tensor of shape [batch_size, seq_len, 2] with input coordinates
            map_features: Tensor of shape [batch_size, map_feature_dim]
            condition_features: Tensor of shape [batch_size, condition_dim]
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
        
        # Expand map and condition features to match sequence length
        map_features_expanded = map_features.unsqueeze(1).expand(-1, seq_len, -1)
        condition_features_expanded = condition_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine LSTM output with context
        combined = torch.cat([lstm_out, map_features_expanded, condition_features_expanded], dim=2)
        context_integrated = self.context_layer(combined)
        
        # Predict next coordinates
        next_coords = self.output_layer(context_integrated)
        
        return next_coords, hidden


class ImprovedSequentialRouteGenerator(nn.Module):
    """Complete model for generating routes sequentially with proper next-coordinate prediction"""
    def __init__(self, map_feature_dim=256, condition_dim=64, hidden_dim=256, num_layers=2):
        super(ImprovedSequentialRouteGenerator, self).__init__()
        
        self.map_encoder = MapEncoder(output_dim=map_feature_dim)
        self.condition_encoder = ConditionEncoder(input_dim=5, output_dim=condition_dim)
        self.route_predictor = RoutePredictor(
            map_feature_dim=map_feature_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
    
    def forward(self, map_img, conditions, input_seq):
        """
        Forward pass through the model for training
        
        Args:
            map_img: The map image tensor [batch_size, channels, height, width]
            conditions: Tensor with distance, start and end points [batch_size, 5]
            input_seq: Input sequence of coordinates [batch_size, seq_len, 2]
            
        Returns:
            predicted_coords: Predicted next coordinates for each input position
        """
        # Encode map and conditions
        map_features = self.map_encoder(map_img)
        condition_features = self.condition_encoder(conditions)
        
        # Predict next coordinates based on input sequence
        predicted_coords, _ = self.route_predictor(input_seq, map_features, condition_features)
        
        return predicted_coords
    
    def generate_route(self, map_img, conditions, max_length=500):
        """
        Generate a complete route autoregressively
        
        Args:
            map_img: The map image tensor [batch_size, channels, height, width]
            conditions: Tensor with distance, start and end points [batch_size, 5]
            max_length: Maximum route length to generate
            
        Returns:
            generated_route: The generated route coordinates
        """
        batch_size = map_img.size(0)
        device = map_img.device
        
        # Encode map and conditions (only done once)
        with torch.no_grad():
            map_features = self.map_encoder(map_img)
            condition_features = self.condition_encoder(conditions)
            
            # Initialize the route with the start point (from conditions)
            start_point = conditions[:, 1:3].unsqueeze(1)  # [batch_size, 1, 2]
            generated_route = [start_point]
            
            # Hidden state for LSTM
            hidden = None
            
            # Generate points one by one autoregressively
            current_seq = start_point
            
            for _ in range(max_length - 1):
                # Predict next coordinate
                next_coord_pred, hidden = self.route_predictor(
                    current_seq, map_features, condition_features, hidden
                )
                
                # Get the last predicted coordinate
                next_coord = next_coord_pred[:, -1:, :]
                
                # Add to the generated route
                generated_route.append(next_coord)
                
                # Update current sequence for next iteration
                # For autoregressive generation, we only use the most recent point
                current_seq = next_coord
            
            # Concatenate all coordinates
            full_route = torch.cat(generated_route, dim=1)
            
        return full_route


def create_padded_batch(batch_data):
    """Create a padded batch for variable-length sequences"""
    # Get data from batch
    images = torch.stack([item['image'] for item in batch_data])
    conditions = torch.stack([item['conditions'] for item in batch_data])
    
    # Get sequences and their lengths
    input_seqs = [item['input_seq'] for item in batch_data]
    target_seqs = [item['target_seq'] for item in batch_data]
    seq_lengths = [item['seq_length'] - 1 for item in batch_data]  # -1 because we're predicting next coords
    
    # Find max sequence length in this batch
    max_len = max(seq_lengths)
    
    # Pad sequences
    padded_inputs = []
    padded_targets = []
    
    for inp, tgt in zip(input_seqs, target_seqs):
        # Pad input sequence
        padded_inp = torch.zeros((max_len, 2))
        padded_inp[:len(inp)] = inp
        padded_inputs.append(padded_inp)
        
        # Pad target sequence
        padded_tgt = torch.zeros((max_len, 2))
        padded_tgt[:len(tgt)] = tgt
        padded_targets.append(padded_tgt)
    
    # Stack into tensors
    padded_inputs = torch.stack(padded_inputs)
    padded_targets = torch.stack(padded_targets)
    seq_lengths = torch.tensor(seq_lengths)
    
    return {
        'image': images,
        'conditions': conditions,
        'input_seq': padded_inputs,
        'target_seq': padded_targets,
        'seq_lengths': seq_lengths
    }


def train_improved_model(model, data_loader, num_epochs=100, lr=0.001, 
                         device='cuda', save_interval=10, save_dir='models'):
    """
    Train the improved sequential route generation model with proper next-step prediction
    """
    print("Initializing training...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Use MSE loss for coordinate regression
    criterion = nn.MSELoss()
    
    # Create log dictionaries
    losses = {'train': []}
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    # Calculate total number of batches
    total_batches = len(data_loader)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        
        # Initialize epoch metrics
        epoch_loss = 0.0
        
        # Progress bar for this epoch
        progress_bar = tqdm(
            enumerate(data_loader), 
            total=total_batches,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=True
        )
        
        for i, batch in progress_bar:
            # Move batch data to device
            images = batch['image'].to(device)
            conditions = batch['conditions'].to(device)
            input_seq = batch['input_seq'].to(device)
            target_seq = batch['target_seq'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predicted_coords = model(images, conditions, input_seq)
            
            # Calculate loss
            loss = criterion(predicted_coords, target_seq)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}"
            })
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / total_batches
        losses['train'].append(avg_epoch_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Calculate estimated time remaining
        avg_epoch_time = (time.time() - start_time) / (epoch + 1)
        epochs_remaining = num_epochs - (epoch + 1)
        est_time_remaining = avg_epoch_time * epochs_remaining
        est_time_str = str(datetime.timedelta(seconds=int(est_time_remaining)))
        
        # Print epoch summary
        print(f"\n{'-'*80}")
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f} seconds")
        print(f"Average loss: {avg_epoch_loss:.4f}")
        print(f"Estimated time remaining: {est_time_str}")
        
        # Save model periodically
        if epoch % save_interval == 0 or epoch == num_epochs - 1:
            print(f"Saving model checkpoint for epoch {epoch+1}...")
            torch.save(model.state_dict(), f"{save_dir}/sequential_model_{epoch}.pt")
            
        # Generate sample routes
        print("Generating sample routes...")
        model.eval()
        with torch.no_grad():
            # Generate routes using a small sample of images from the batch
            num_samples = min(4, images.size(0))
            sample_maps = images[:num_samples]
            sample_conditions = conditions[:num_samples]
            
            # Generate routes
            sample_routes = model.generate_route(sample_maps, sample_conditions)
            
            # Visualize and save
            sample_path = f"{save_dir}/sample_epoch_{epoch}.png"
            visualize_routes(sample_maps, sample_routes, sample_path)
            print(f"Sample routes saved to {sample_path}")
    
    # Calculate total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print(f"{'-'*80}")
    print(f"Training completed in {total_time_str}")
    print(f"Final average loss: {avg_epoch_loss:.4f}")
    
    return model, losses


def visualize_routes(maps, routes, save_path=None):
    """
    Visualize generated routes on maps
    """
    fig, axes = plt.subplots(len(routes), 1, figsize=(10, 5 * len(routes)))
    
    if len(routes) == 1:
        axes = [axes]
    
    for i, (map_img, route) in enumerate(zip(maps, routes)):
        # Convert map from tensor to numpy for visualization
        map_np = map_img.cpu().permute(1, 2, 0).numpy()
        # Denormalize if necessary (assuming maps are normalized to [-1, 1])
        map_np = (map_np * 0.5) + 0.5
        map_np = np.clip(map_np, 0, 1)
        
        # Convert route from tensor to numpy of integers
        route_np = route.cpu().numpy()
        route_np = route_np.astype(int)
        
        # Display the map
        axes[i].imshow(map_np)
        
        # Overlay the route
        axes[i].plot(route_np[:, 0], route_np[:, 1], 'r-', linewidth=1)
        
        # Highlight start and end points
        axes[i].plot(route_np[0, 0], route_np[0, 1], 'go', markersize=8)  # Start: green
        axes[i].plot(route_np[-1, 0], route_np[-1, 1], 'bo', markersize=8)  # End: blue
        
        axes[i].set_title(f"Generated Route {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main():
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    print(f"{'-'*80}")
    
    # Configuration
    data_path = "data/processed_combined.csv"
    img_dir1 = "image_data/images0_25000"
    img_dir2 = "image_data"
    batch_size = 16
    num_epochs = 50
    
    print(f"Configuration:")
    print(f"- Data: {data_path}")
    print(f"- Batch size: {batch_size}")
    print(f"- Epochs: {num_epochs}")
    print(f"{'-'*80}")
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((683, 1366)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create dataset
    dataset = RunningRouteDataset(data_path, [img_dir1, img_dir2], transform=transform)
    
    print(f"Creating data loader with {len(dataset)} samples")
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=create_padded_batch  # Handle variable-length sequences
    )
    
    # Initialize model
    model = ImprovedSequentialRouteGenerator().to(device)
    print("Model created successfully")
    
    # Train model
    print(f"{'-'*80}")
    trained_model, losses = train_improved_model(
        model,
        dataloader, 
        num_epochs=num_epochs, 
        device=device,
        save_dir='models',
        save_interval=5
    )
    
    # Save final model
    print("Saving final model...")
    torch.save(trained_model.state_dict(), "models/sequential_model_final.pt")
    
    # Plot losses
    print("Generating loss plot...")
    plt.figure(figsize=(10, 5))
    plt.plot(losses['train'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    loss_plot_path = "models/training_losses.png"
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")
    
    print(f"{'-'*80}")
    print("Training complete!")


if __name__ == "__main__":
    main()