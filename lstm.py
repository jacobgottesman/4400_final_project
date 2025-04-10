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
import torch.multiprocessing as mp
from functools import partial
import numpy as np
import concurrent.futures

class RunningRouteDataset(Dataset):
    def __init__(self, csv_path, img_dirs, transform=None, verbose=False, local_map_size=128):
        """
        Dataset for running routes and map images
        
        Args:
            csv_path: Path to CSV file with metadata and route information
            img_dirs: List of directories containing map images
            transform: Transforms to apply to images
            verbose: Whether to print verbose logs during data loading
            local_map_size: Size of the local map crop around current point
        """
        print(f"Loading dataset from {csv_path}...")
        self.data = pd.read_csv(csv_path)
        # self.data = self.data.iloc[0:1000, :]
        self.data['route_xy'] = self.data['route_xy'].apply(eval)
        self.img_dirs = img_dirs
        self.transform = transform
        self.verbose = verbose
        self.local_map_size = local_map_size
        
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
    
    def load_full_map(self, img_id):
        """Load the full map image for a given image ID"""
        if img_id > 50425:
            img_path = os.path.join(self.img_dirs[1], f"map{img_id}.png")
        else:
            img_path = os.path.join(self.img_dirs[0], f"map{img_id}.png")
            
        return Image.open(img_path).convert('RGB')
    
    def __getitem__(self, idx):
        # Get the row from the dataframe
        row = self.data.iloc[idx]
        
        # Load image
        img_id = row['index']
        
        if self.verbose:
            print(f"Loading image for index {idx}, Image ID: {img_id}")
        
        # Load the full map image (we'll use this for reference and local cropping)
        full_image = self.load_full_map(img_id)
        
        # Get route coordinates
        route_xy = np.array(eval(row['route_xy'])) if isinstance(row['route_xy'], str) else row['route_xy']
        route_tensor = torch.tensor(route_xy, dtype=torch.float32)
        
        # Get conditional parameters
        distance = torch.tensor([row['distance']], dtype=torch.float32)
        
        # Extract start and end points
        start_point = torch.tensor(route_xy[0], dtype=torch.float32)
        end_point = torch.tensor(route_xy[-1], dtype=torch.float32)
        
        # Store the full image with transform applied
        if self.transform:
            transformed_full_image = self.transform(full_image)
        else:
            transformed_full_image = transforms.ToTensor()(full_image)
        
        # Create input-target pairs for sequence learning
        # Input: all coordinates except the last one
        # Target: all coordinates except the first one
        input_seq = route_tensor[:-1]
        target_seq = route_tensor[1:]
        
        # Calculate steps remaining for each position in the sequence
        # This helps the model know how many steps until it should reach the end
        seq_length = len(route_xy)
        steps_remaining = torch.arange(seq_length-1, 0, -1).float()
        
        return {
            'full_image': transformed_full_image,
            'route': route_tensor,
            'input_seq': input_seq,
            'target_seq': target_seq,
            'conditions': torch.cat([distance, start_point, end_point], dim=0),
            'seq_length': seq_length,
            'steps_remaining': steps_remaining,
            'img_id': img_id,
            'original_img_size': torch.tensor([full_image.height, full_image.width])
        }


class LocalMapEncoder(nn.Module):
    """Encodes local map crops around the current position"""
    def __init__(self, output_dim=256, local_map_size=128):
        super(LocalMapEncoder, self).__init__()
        
        self.local_map_size = local_map_size
        
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
        
        # Calculate the flattened size based on local map size
        # After 4 stride-2 layers: size = original_size / 16
        conv_output_size = local_map_size // 16
        self.flattened_size = 128 * conv_output_size * conv_output_size
        
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


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


class ImprovedRoutePredictor(nn.Module):
    """LSTM model for predicting the next coordinate based on previous coordinates
    and local map features"""
    def __init__(self, map_feature_dim=256, condition_dim=64, hidden_dim=256, num_layers=2):
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
            map_features: Tensor of shape [batch_size, seq_len, map_feature_dim]
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
        
        # Combine LSTM output with context (map features and conditions)
        combined = torch.cat([lstm_out, map_features, condition_features], dim=2)
        context_integrated = self.context_layer(combined)
        
        # Predict next coordinates
        next_coords = self.output_layer(context_integrated)
        
        return next_coords, hidden


class LocalMapAwareRouteGenerator(nn.Module):
    """Enhanced model with local map awareness and steps remaining information"""
    def __init__(self, map_feature_dim=256, condition_dim=64, hidden_dim=256, 
                 num_layers=2, local_map_size=128):
        super(LocalMapAwareRouteGenerator, self).__init__()
        
        self.local_map_size = local_map_size
        self.map_feature_dim = map_feature_dim
        
        # Local map encoder instead of global map encoder
        self.local_map_encoder = LocalMapEncoder(
            output_dim=map_feature_dim,
            local_map_size=local_map_size
        )
        
        # Enhanced condition encoder that includes steps remaining
        self.condition_encoder = ConditionEncoder(
            input_dim=6,  # distance, start_x, start_y, end_x, end_y, steps_remaining
            output_dim=condition_dim
        )
        
        # Improved route predictor
        self.route_predictor = ImprovedRoutePredictor(
            map_feature_dim=map_feature_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )


    def get_local_map_crops(self, full_map, coords, map_size):
        """
        Extract local map crops with optimized NumPy operations
        """
        import numpy as np
        
        # Move tensors to CPU and convert to numpy
        full_map_np = full_map.cpu().numpy()
        coords_np = coords.cpu().numpy()
        
        batch_size, seq_len, _ = coords_np.shape
        channels = full_map_np.shape[1]
        h, w = full_map_np.shape[2:4]
        half_size = map_size // 2
        
        # Pre-allocate output array
        result_np = np.zeros((batch_size, seq_len, channels, map_size, map_size), dtype=full_map_np.dtype)
        
        # Process batches sequentially, but use vectorized operations within batch
        for batch_idx in range(batch_size):
            y_coords = coords_np[batch_idx, :, 0].astype(np.int32)
            x_coords = coords_np[batch_idx, :, 1].astype(np.int32)
            
            # Calculate all boundaries at once
            y_min = np.maximum(0, y_coords - half_size)
            x_min = np.maximum(0, x_coords - half_size)
            y_max = np.minimum(h, y_coords + half_size)
            x_max = np.minimum(w, x_coords + half_size)
            
            # Get each crop and apply padding if needed
            for seq_idx in range(seq_len):
                # Extract crop
                crop = full_map_np[batch_idx, :, y_min[seq_idx]:y_max[seq_idx], x_min[seq_idx]:x_max[seq_idx]]
                
                # Calculate padding if needed
                pad_top = max(0, half_size - y_coords[seq_idx])
                pad_left = max(0, half_size - x_coords[seq_idx])
                pad_bottom = max(0, y_coords[seq_idx] + half_size - h)
                pad_right = max(0, x_coords[seq_idx] + half_size - w)
                
                # Apply padding if needed
                if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
                    padded_crop = np.zeros((channels, map_size, map_size), dtype=crop.dtype)
                    crop_h, crop_w = crop.shape[1:3]
                    padded_crop[:, pad_top:pad_top+crop_h, pad_left:pad_left+crop_w] = crop
                    crop = padded_crop
                
                # Store in result array
                result_np[batch_idx, seq_idx] = crop
        
        # Convert result back to PyTorch tensor and move to original device
        result_tensor = torch.from_numpy(result_np).to(full_map.device)
        
        return result_tensor

   
    def forward(self, full_map, conditions, input_seq, steps_remaining):
        """
        Forward pass through the model for training
        
        Args:
            full_map: The full map image tensor [batch_size, channels, height, width]
            conditions: Tensor with distance, start and end points [batch_size, 5]
            input_seq: Input sequence of coordinates [batch_size, seq_len, 2]
            steps_remaining: Steps remaining to end for each position [batch_size, seq_len]
            
        Returns:
            predicted_coords: Predicted next coordinates for each input position
        """
        batch_size, seq_len, _ = input_seq.shape
        device = full_map.device
        
        # Get local map crops around each input coordinate
        local_maps = self.get_local_map_crops(full_map, input_seq, self.local_map_size)
        
        # Reshape for processing
        local_maps_flat = local_maps.reshape(batch_size * seq_len, 3, self.local_map_size, self.local_map_size)
        
        # Encode each local map
        local_map_features = self.local_map_encoder(local_maps_flat)
        
        # Reshape back to [batch_size, seq_len, feature_dim]
        map_features = local_map_features.reshape(batch_size, seq_len, self.map_feature_dim)
        
        # Enhance conditions with steps remaining
        enhanced_conditions = []
        for b in range(batch_size):
            # Combine base conditions with steps remaining for each position
            base_cond = conditions[b].unsqueeze(0).expand(seq_len, -1)  # [seq_len, 5]
            steps = steps_remaining[b, :seq_len].unsqueeze(1)  # [seq_len, 1]
            combined = torch.cat([base_cond, steps], dim=1)  # [seq_len, 6]
            enhanced_conditions.append(combined)
        
        enhanced_conditions = torch.stack(enhanced_conditions)  # [batch_size, seq_len, 6]
        
        # Encode enhanced conditions
        condition_features = self.condition_encoder(enhanced_conditions)
        
        # Predict next coordinates
        predicted_coords, _ = self.route_predictor(
            input_seq, 
            map_features, 
            condition_features
        )
        
        return predicted_coords
    
    def generate_route(self, full_map, conditions, max_length=500):
        """
        Generate a complete route autoregressively
        
        Args:
            full_map: The full map image tensor [batch_size, channels, height, width]
            conditions: Tensor with distance, start and end points [batch_size, 5]
            max_length: Maximum route length to generate
            
        Returns:
            generated_route: The generated route coordinates
        """
        batch_size = full_map.size(0)
        device = full_map.device
        
        with torch.no_grad():            
            # Initialize the route with the start point (from conditions)
            start_point = conditions[:, 1:3].unsqueeze(1)  # [batch_size, 1, 2]
            generated_route = [start_point]
            
            # Hidden state for LSTM
            hidden = None
            
            # Generate points one by one autoregressively
            current_point = start_point
            
            for step in range(max_length - 1):
                # Calculate steps remaining
                steps_remaining = torch.tensor(
                    [[max_length - step - 1]], 
                    dtype=torch.float32, 
                    device=device
                ).expand(batch_size, 1)
                
                # Get local map around current point
                local_map = self.get_local_map_crops(
                    full_map, 
                    current_point, 
                    self.local_map_size
                )
                
                # Reshape for processing (batch_size, 1, C, H, W) -> (batch_size, C, H, W)
                local_map = local_map.squeeze(1)
                
                # Encode local map
                map_features = self.local_map_encoder(local_map).unsqueeze(1)  # [batch_size, 1, feature_dim]
                
                # Enhance conditions with steps remaining
                enhanced_conditions = torch.cat([
                    conditions, 
                    steps_remaining
                ], dim=1).unsqueeze(1)  # [batch_size, 1, 6]
                
                # Encode conditions
                condition_features = self.condition_encoder(enhanced_conditions)
                
                # Predict next coordinate
                next_coord_pred, hidden = self.route_predictor(
                    current_point, 
                    map_features, 
                    condition_features, 
                    hidden
                )
                
                # Get the predicted coordinate
                next_point = next_coord_pred
                
                # Add to the generated route
                generated_route.append(next_point)
                
                # Update current point for next iteration
                current_point = next_point
                
                # Check if we're close enough to the end point
                end_points = conditions[:, 3:5].unsqueeze(1)  # [batch_size, 1, 2]
                distances_to_end = torch.norm(current_point - end_points, dim=2)
                
                # If all routes are close to their end points, we can stop early
                if torch.all(distances_to_end < 10):  # 10 pixels threshold
                    break
            
            # Concatenate all coordinates
            full_route = torch.cat(generated_route, dim=1)
            
        return full_route


def create_padded_batch(batch_data):
    """Create a padded batch for variable-length sequences"""
    # Get data from batch
    images = torch.stack([item['full_image'] for item in batch_data])
    conditions = torch.stack([item['conditions'] for item in batch_data])
    
    # Get sequences and their lengths
    input_seqs = [item['input_seq'] for item in batch_data]
    target_seqs = [item['target_seq'] for item in batch_data]
    steps_remaining = [item['steps_remaining'] for item in batch_data]
    seq_lengths = [item['seq_length'] - 1 for item in batch_data]  # -1 because we're predicting next coords
    
    # Find max sequence length in this batch
    max_len = max(seq_lengths)
    
    # Pad sequences
    padded_inputs = []
    padded_targets = []
    padded_steps = []
    
    for inp, tgt, steps in zip(input_seqs, target_seqs, steps_remaining):
        # Pad input sequence
        padded_inp = torch.zeros((max_len, 2))
        padded_inp[:len(inp)] = inp
        padded_inputs.append(padded_inp)
        
        # Pad target sequence
        padded_tgt = torch.zeros((max_len, 2))
        padded_tgt[:len(tgt)] = tgt
        padded_targets.append(padded_tgt)
        
        # Pad steps remaining
        padded_step = torch.zeros(max_len)
        padded_step[:len(steps)] = steps
        padded_steps.append(padded_step)
    
    # Stack into tensors
    padded_inputs = torch.stack(padded_inputs)
    padded_targets = torch.stack(padded_targets)
    padded_steps = torch.stack(padded_steps)
    seq_lengths = torch.tensor(seq_lengths)
    
    return {
        'full_image': images,
        'conditions': conditions,
        'input_seq': padded_inputs,
        'target_seq': padded_targets,
        'steps_remaining': padded_steps,
        'seq_lengths': seq_lengths
    }


def train_local_map_model(model, data_loader, num_epochs=100, lr=0.001, 
                         device='cuda', save_interval=10, save_dir='models/lstm',
                         log_dir='tensorboard_logs'):
    """
    Train the local map aware route generation model with step-remaining tracking
    """
    print("Initializing training...")
    
    # Import TensorBoard modules
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np
    
    # Create TensorBoard writer
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, run_id)
    writer = SummaryWriter(log_path)
    print(f"TensorBoard logs will be saved to {log_path}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Use MSE loss for coordinate regression
    coord_criterion = nn.MSELoss()
    
    # Create log dictionaries
    losses = {'train': []}
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    # Calculate total number of batches
    total_batches = len(data_loader)
    
    # Global step counter for TensorBoard
    global_step = 0
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        
        # Initialize epoch metrics
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_mae = 0.0
        epoch_distance_error = 0.0
        
        # Progress bar for this epoch
        progress_bar = tqdm(
            enumerate(data_loader), 
            total=total_batches,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=True
        )
        
        for i, batch in progress_bar:
            # Move batch data to device
            full_images = batch['full_image'].to(device)
            conditions = batch['conditions'].to(device)
            input_seq = batch['input_seq'].to(device)
            target_seq = batch['target_seq'].to(device)
            steps_remaining = batch['steps_remaining'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predicted_coords = model(full_images, conditions, input_seq, steps_remaining)
            
            # Calculate primary loss (MSE)
            loss = coord_criterion(predicted_coords, target_seq)
            
            # Calculate additional metrics for logging
            with torch.no_grad():
                # Mean Absolute Error
                mae = torch.mean(torch.abs(predicted_coords - target_seq))
                
                # Euclidean distance error (for each point pair)
                pred_reshaped = predicted_coords.view(-1, 2)
                target_reshaped = target_seq.view(-1, 2)
                distance_error = torch.sqrt(torch.sum((pred_reshaped - target_reshaped)**2, dim=1)).mean()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_mse += loss.item()
            epoch_mae += mae.item()
            epoch_distance_error += distance_error.item()
            
            # Log batch metrics to TensorBoard
            writer.add_scalar('Batch/Loss', loss.item(), global_step)
            writer.add_scalar('Batch/MAE', mae.item(), global_step)
            writer.add_scalar('Batch/Distance_Error', distance_error.item(), global_step)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dist_err': f"{distance_error.item():.4f}"
            })
            
            global_step += 1
        
        # Calculate average epoch metrics
        avg_epoch_loss = epoch_loss / total_batches
        avg_epoch_mse = epoch_mse / total_batches
        avg_epoch_mae = epoch_mae / total_batches
        avg_epoch_distance_error = epoch_distance_error / total_batches
        
        losses['train'].append(avg_epoch_loss)
        
        # Log epoch metrics to TensorBoard
        writer.add_scalar('Epoch/Loss', avg_epoch_loss, epoch)
        writer.add_scalar('Epoch/MSE', avg_epoch_mse, epoch)
        writer.add_scalar('Epoch/MAE', avg_epoch_mae, epoch)
        writer.add_scalar('Epoch/Distance_Error', avg_epoch_distance_error, epoch)
        
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
        print(f"Average loss: {avg_epoch_loss:.4f}, Distance error: {avg_epoch_distance_error:.4f}")
        print(f"Estimated time remaining: {est_time_str}")
        
        # Save model periodically
        if epoch % save_interval == 0 or epoch == num_epochs - 1:
            print(f"Saving model checkpoint for epoch {epoch+1}...")
            model_path = f"{save_dir}/local_map_model_epoch_{epoch}.pt"
            torch.save(model.state_dict(), model_path)
            
            # Add model checkpoint to TensorBoard
            writer.add_text('Checkpoints', f"Model saved at epoch {epoch+1}: {model_path}", epoch)
            
        # Generate sample routes
        print("Generating sample routes...")
        model.eval()
        with torch.no_grad():
            # Generate routes using a small sample of images from the batch
            num_samples = min(4, full_images.size(0))
            sample_maps = full_images[:num_samples]
            sample_conditions = conditions[:num_samples]
            
            # Generate routes
            sample_routes = model.generate_route(sample_maps, sample_conditions)
            
            # Visualize and save
            sample_path = f"{save_dir}/sample_local_map_epoch_{epoch}.png"
            visualize_routes(sample_maps, sample_routes, sample_path)
            print(f"Sample routes saved to {sample_path}")
            
            # Add sample route images to TensorBoard
            try:
                sample_img = plt.imread(sample_path)
                writer.add_image('Generated Routes', sample_img.transpose(2, 0, 1), epoch, dataformats='CHW')
            except Exception as e:
                print(f"Failed to log route images to TensorBoard: {e}")
    
    # Calculate total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print(f"{'-'*80}")
    print(f"Training completed in {total_time_str}")
    print(f"Final average loss: {avg_epoch_loss:.4f}")
    print(f"TensorBoard logs saved to {log_path}")
    
    # Close TensorBoard writer
    writer.close()
    
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
        axes[i].plot(route_np[:, 1], route_np[:, 0], 'r-', linewidth=1)
        
        # Highlight start and end points
        axes[i].plot(route_np[0, 1], route_np[0, 0], 'go', markersize=8)  # Start: green
        axes[i].plot(route_np[-1, 1], route_np[-1, 0], 'bo', markersize=8)  # End: blue
        
        axes[i].set_title(f"Generated Route {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

import torch
import time
import functools
import cProfile
import pstats
import io
from torch.utils.data import DataLoader
from memory_profiler import profile as memory_profile
from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda.amp import autocast, GradScaler

class PerformanceProfiler:
    """Utility class to profile different parts of the ML pipeline"""
    
    @staticmethod
    def measure_time(func):
        """Decorator to measure execution time of a function"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute")
            return result
        return wrapper
    
    @staticmethod
    def profile_function(func):
        """Decorator to profile a function using cProfile"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            result = func(*args, **kwargs)
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Print top 20 time-consuming functions
            print(s.getvalue())
            return result
        return wrapper
    
    @staticmethod
    def profile_memory(func):
        """Decorator to profile memory usage of a function"""
        @memory_profile
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    
    @staticmethod
    def profile_data_loading(dataset, batch_size=16, num_workers=0):
        """Profile data loading performance"""
        print(f"Profiling DataLoader with batch_size={batch_size}, num_workers={num_workers}")
        
        # Create data loader
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            collate_fn=create_padded_batch
        )
        
        # Time data loading
        start_time = time.time()
        for i, batch in enumerate(loader):
            if i >= 10:  # Only test a few batches
                break
        end_time = time.time()
        
        avg_time = (end_time - start_time) / min(10, len(loader))
        print(f"Average batch loading time: {avg_time:.4f} seconds")
        
        return avg_time
    
    @staticmethod
    def profile_gpu_operations(model, dataloader, device, num_batches=5):
        """Profile GPU operations using PyTorch profiler"""
        model = model.to(device)
        
        # Warm-up
        print("Warming up...")
        for i, batch in enumerate(dataloader):
            if i >= 3:
                break
            
            full_images = batch['full_image'].to(device)
            conditions = batch['conditions'].to(device)
            input_seq = batch['input_seq'].to(device)
            steps_remaining = batch['steps_remaining'].to(device)
            
            with torch.no_grad():
                _ = model(full_images, conditions, input_seq, steps_remaining)
        
        print("Starting GPU profiling...")
        # Profile with PyTorch profiler
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True
        ) as prof:
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                with record_function("batch_processing"):
                    full_images = batch['full_image'].to(device)
                    conditions = batch['conditions'].to(device)
                    input_seq = batch['input_seq'].to(device)
                    target_seq = batch['target_seq'].to(device)
                    steps_remaining = batch['steps_remaining'].to(device)
                    
                    with record_function("forward_pass"):
                        predicted_coords = model(full_images, conditions, input_seq, steps_remaining)
                    
                    with record_function("loss_calculation"):
                        loss = torch.nn.functional.mse_loss(predicted_coords, target_seq)
                    
                    with record_function("backward_pass"):
                        loss.backward()
        
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        return prof
    
    @staticmethod
    def profile_specific_components(model, batch, device):
        """Profile specific components of the model to identify bottlenecks"""
        # Move data to device
        full_images = batch['full_image'].to(device)
        conditions = batch['conditions'].to(device)
        input_seq = batch['input_seq'].to(device)
        steps_remaining = batch['steps_remaining'].to(device)
        
        batch_size, seq_len, _ = input_seq.shape
        
        # Profile get_local_map_crops
        start_time = time.time()
        with record_function("get_local_map_crops"):
            local_maps = model.get_local_map_crops(full_images, input_seq, model.local_map_size)
        crop_time = time.time() - start_time
        print(f"get_local_map_crops took {crop_time:.4f} seconds")
        
        # Profile local_map_encoder
        local_maps_flat = local_maps.reshape(batch_size * seq_len, 3, model.local_map_size, model.local_map_size)
        start_time = time.time()
        with record_function("local_map_encoder"):
            local_map_features = model.local_map_encoder(local_maps_flat)
        encoder_time = time.time() - start_time
        print(f"local_map_encoder took {encoder_time:.4f} seconds")
        
        # Profile condition_encoder
        # Prepare enhanced conditions
        enhanced_conditions = []
        for b in range(batch_size):
            base_cond = conditions[b].unsqueeze(0).expand(seq_len, -1)
            steps = steps_remaining[b, :seq_len].unsqueeze(1)
            combined = torch.cat([base_cond, steps], dim=1)
            enhanced_conditions.append(combined)
        enhanced_conditions = torch.stack(enhanced_conditions)
        
        start_time = time.time()
        with record_function("condition_encoder"):
            condition_features = model.condition_encoder(enhanced_conditions)
        condition_time = time.time() - start_time
        print(f"condition_encoder took {condition_time:.4f} seconds")
        
        # Profile route_predictor
        map_features = local_map_features.reshape(batch_size, seq_len, model.map_feature_dim)
        start_time = time.time()
        with record_function("route_predictor"):
            predicted_coords, _ = model.route_predictor(input_seq, map_features, condition_features)
        predictor_time = time.time() - start_time
        print(f"route_predictor took {predictor_time:.4f} seconds")
        
        return {
            "get_local_map_crops": crop_time,
            "local_map_encoder": encoder_time,
            "condition_encoder": condition_time,
            "route_predictor": predictor_time
        }


def profile_main():
    """Main function to profile the model"""
    # Load configuration
    print("Setting up profiling environment...")
    data_path = "data/processed_combined.csv"
    img_dir1 = "image_data/images0_25000"
    img_dir2 = "image_data"
    batch_size = 16
    
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Create dataset with limited size for profiling
    transform = transforms.Compose([
        transforms.Resize((683, 1366)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create dataset with verbose=True to see data loading details
    dataset = RunningRouteDataset(
        data_path, 
        [img_dir1, img_dir2], 
        transform=transform, 
        verbose=True
    )
    
    print(f"Created dataset with {len(dataset)} samples")
    
    # Profile data loading with different num_workers
    print("\n=== Profiling DataLoader Performance ===")
    worker_times = {}
    for workers in [0, 2, 4, 8]:
        if workers > 0 and device == torch.device("mps"):
            print(f"Skipping num_workers={workers} for MPS device (not supported)")
            continue
        time_taken = PerformanceProfiler.profile_data_loading(
            dataset, 
            batch_size=batch_size, 
            num_workers=workers
        )
        worker_times[workers] = time_taken
    
    optimal_workers = min(worker_times, key=worker_times.get)
    print(f"Optimal num_workers setting: {optimal_workers}")
    
    # Create dataloader with optimal workers
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=optimal_workers,
        collate_fn=create_padded_batch
    )
    
    # Initialize model
    print("\n=== Initializing Model ===")
    model = LocalMapAwareRouteGenerator().to(device)
    
    # Get a batch for detailed profiling
    for batch in dataloader:
        break
    
    # Profile model components
    print("\n=== Profiling Model Components ===")
    component_times = PerformanceProfiler.profile_specific_components(model, batch, device)
    
    # Sort components by time
    sorted_components = sorted(component_times.items(), key=lambda x: x[1], reverse=True)
    print("\nComponent timing summary (slowest to fastest):")
    for component, time_taken in sorted_components:
        print(f"{component}: {time_taken:.4f} seconds ({time_taken/sum(component_times.values())*100:.1f}%)")
    
    # Profile GPU operations
    if device.type == "cuda":
        print("\n=== Profiling GPU Operations ===")
        gpu_profile = PerformanceProfiler.profile_gpu_operations(model, dataloader, device)
    
    # Profile a training step
    print("\n=== Profiling Complete Training Step ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    @PerformanceProfiler.profile_function
    def train_step(model, batch, optimizer, device):
        # Move batch data to device
        full_images = batch['full_image'].to(device)
        conditions = batch['conditions'].to(device)
        input_seq = batch['input_seq'].to(device)
        target_seq = batch['target_seq'].to(device)
        steps_remaining = batch['steps_remaining'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predicted_coords = model(full_images, conditions, input_seq, steps_remaining)
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(predicted_coords, target_seq)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return loss.item()
    
    loss = train_step(model, batch, optimizer, device)
    print(f"Training step loss: {loss:.4f}")
    
    print("\n=== Profiling Complete ===")
    return {
        "dataset": dataset,
        "dataloader": dataloader,
        "model": model,
        "device": device,
        "worker_times": worker_times,
        "component_times": component_times
    }

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
    model = LocalMapAwareRouteGenerator().to(device)
    print("Model created successfully")

    # Train model
    print(f"{'-'*80}")
    trained_model, losses = train_local_map_model(
        model,
        dataloader, 
        num_epochs=num_epochs, 
        device=device,
        save_dir='models/lstm',
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

    profile_main()
    results = main()