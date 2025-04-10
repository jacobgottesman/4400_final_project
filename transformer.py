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

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the token embeddings to introduce a notion of word order.
    """
    def __init__(self, d_model, max_seq_length=500, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MapImageEncoder(nn.Module):
    """
    Encodes the map image into a feature representation using a CNN backbone.
    """
    def __init__(self, output_dim=512):
        super(MapImageEncoder, self).__init__()
        
        # CNN backbone for feature extraction
        self.conv_layers = nn.Sequential(
            # Layer 1: 3 -> 32 channels, 1366x683 -> 341x170
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Layer 2: 32 -> 64 channels, 341x170 -> 113x56
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 3: 64 -> 128 channels, 113x56 -> 56x28
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Layer 4: 128 -> 256 channels, 56x28 -> 28x14
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Calculate flattened size after CNN layers (28 x 14 x 256)
        self.flattened_size = 28 * 14 * 256
        
        # Projection to desired output dimension
        self.projection = nn.Sequential(
            nn.Linear(self.flattened_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim)
        )
        
    def forward(self, x):
        # Extract features with CNN
        x = self.conv_layers(x)
        
        # Flatten and project
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return x


class SpatialAttention(nn.Module):
    """
    Extracts spatially aware features from the map by using attention 
    to focus on relevant map regions.
    """
    def __init__(self, feature_dim=256, num_heads=8):
        super(SpatialAttention, self).__init__()
        
        # Self-attention layer to capture spatial relationships
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=num_heads,
            dropout=0.1
        )
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(feature_dim)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(0.1)
        )
        
        # Final normalization
        self.final_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        # Self-attention (batch_size, seq_len, feature_dim)
        # Reshape for attention: (seq_len, batch_size, feature_dim)
        x_reshaped = x.permute(1, 0, 2)
        
        # Apply self-attention
        attended, _ = self.self_attention(x_reshaped, x_reshaped, x_reshaped)
        
        # Skip connection and normalization
        attended = attended.permute(1, 0, 2)  # Back to (batch_size, seq_len, feature_dim)
        attended = self.norm(x + attended)
        
        # Feed-forward with skip connection
        ff_out = self.feed_forward(attended)
        out = self.final_norm(attended + ff_out)
        
        return out


class ConditionEncoder(nn.Module):
    """
    Encodes conditional information (distance, duration, etc.) into features.
    """
    def __init__(self, input_dim=4, output_dim=128):
        super(ConditionEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


class RouteCoordinateEmbedding(nn.Module):
    """
    Embeds route coordinates (x, y) into a higher-dimensional representation.
    """
    def __init__(self, embedding_dim=128):
        super(RouteCoordinateEmbedding, self).__init__()
        
        self.embedding = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
    
    def forward(self, coords):
        # Input: (batch_size, seq_len, 2) -> Output: (batch_size, seq_len, embedding_dim)
        batch_size, seq_len, _ = coords.shape
        coords_flat = coords.view(-1, 2)
        embedded = self.embedding(coords_flat)
        return embedded.view(batch_size, seq_len, -1)


class TransformerRouteDecoder(nn.Module):
    """
    Transformer decoder for generating route coordinates autoregressively.
    """
    def __init__(self, 
                 d_model=128, 
                 nhead=8, 
                 num_layers=6, 
                 dim_feedforward=512, 
                 dropout=0.1):
        super(TransformerRouteDecoder, self).__init__()
        
        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Use batch_first=True for modern PyTorch
        )
        
        # Stack multiple decoder layers
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection to coordinate space
        self.output_projection = nn.Linear(d_model, 2)
        
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Forward pass through the decoder.
        
        Args:
            tgt: Target sequence (embedded route coordinates)
            memory: Memory from encoder (map and condition features)
            tgt_mask: Target mask for causal attention
            tgt_key_padding_mask: Mask for padded positions
            
        Returns:
            Predicted coordinates
        """
        # Pass through transformer decoder
        decoder_output = self.decoder(
            tgt, 
            memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project to coordinate space
        predicted_coords = self.output_projection(decoder_output)
        
        return predicted_coords


class RouteTransformer(nn.Module):
    """
    Complete Transformer model for generating routes based on map images and conditions.
    """
    def __init__(self, 
                 d_model=128, 
                 nhead=8, 
                 num_encoder_layers=3,
                 num_decoder_layers=6, 
                 dim_feedforward=512,
                 dropout=0.1,
                 max_route_length=500):
        super(RouteTransformer, self).__init__()
        
        # Map encoding
        self.map_encoder = MapImageEncoder(output_dim=d_model)
        
        # Spatial attention for map features
        self.spatial_attention = SpatialAttention(feature_dim=d_model, num_heads=nhead)
        
        # Condition encoding
        self.condition_encoder = ConditionEncoder(input_dim=4, output_dim=d_model)
        
        # Route coordinate embedding
        self.coord_embedding = RouteCoordinateEmbedding(embedding_dim=d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_seq_length=max_route_length)
        
        # Transformer decoder
        self.transformer_decoder = TransformerRouteDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Store model dimension
        self.d_model = d_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _create_causal_mask(self, size):
        """
        Creates a causal mask for the transformer decoder to prevent attending to future positions.
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.to(self.device)
    
    def _prepare_encoder_memory(self, map_img, conditions):
        """
        Prepares the encoder memory for the transformer decoder.
        Combines map features and condition features.
        """
        # Encode map
        map_features = self.map_encoder(map_img)
        
        # Prepare map features for the transformer (add sequence dimension)
        map_features = map_features.unsqueeze(1)
        
        # Apply spatial attention to map features
        map_features = self.spatial_attention(map_features)
        
        # Encode conditions
        condition_features = self.condition_encoder(conditions).unsqueeze(1)
        
        # Combine map and condition features for the encoder memory
        memory = torch.cat([map_features, condition_features], dim=1)
        
        return memory
        
    def forward(self, map_img, conditions, input_coords, target_mask=None):
        """
        Forward pass for training.
        
        Args:
            map_img: Map image tensor
            conditions: Condition tensor (distance, duration, etc.)
            input_coords: Input route coordinates (for teacher forcing)
            target_mask: Optional mask for handling padded sequences
            
        Returns:
            Predicted coordinates
        """
        # Prepare encoder output as memory
        memory = self._prepare_encoder_memory(map_img, conditions)
        
        # Embed input coordinates
        embedded_coords = self.coord_embedding(input_coords)
        
        # Add positional encoding
        embedded_coords = self.positional_encoding(embedded_coords)
        
        # Create causal mask if not provided
        if target_mask is None:
            target_mask = self._create_causal_mask(input_coords.size(1))
        
        # Generate predictions with transformer decoder
        predicted_coords = self.transformer_decoder(
            embedded_coords, 
            memory,
            tgt_mask=target_mask
        )
        
        return predicted_coords
    
    def generate_route(self, map_img, conditions, start_point=None, max_length=500, temperature=1.0):
        """
        Generates a route autoregressively.
        
        Args:
            map_img: Map image tensor
            conditions: Condition tensor (distance, duration, etc.)
            start_point: Optional starting point, otherwise use center of image
            max_length: Maximum route length to generate
            temperature: Temperature for sampling (higher = more random)
            
        Returns:
            Generated route coordinates as tensor
        """
        batch_size = map_img.size(0)
        device = map_img.device
        
        # Prepare encoder memory (only done once)
        with torch.no_grad():
            memory = self._prepare_encoder_memory(map_img, conditions)
            
            # Initialize route with start point or center of image
            if start_point is None:
                # Use center of image as starting point
                start_point = torch.tensor([[0.5, 0.5]], dtype=torch.float32).to(device)
                start_point = start_point.repeat(batch_size, 1, 1)  # [batch_size, 1, 2]
            else:
                # Make sure start_point has the right shape
                start_point = start_point.unsqueeze(1) if start_point.dim() == 2 else start_point  # [batch_size, 1, 2]
            
            # Initialize the sequence of generated points
            generated_points = [start_point]
            current_seq = start_point
            
            # Generate points autoregressively
            for step in range(max_length - 1):
                # Embed the current sequence
                embedded_seq = self.coord_embedding(current_seq)
                
                # Add positional encoding
                embedded_seq = self.positional_encoding(embedded_seq)
                
                # Get next coordinate prediction
                with torch.no_grad():
                    # If sequence length > 1, we need a causal mask
                    mask = self._create_causal_mask(current_seq.size(1)) if current_seq.size(1) > 1 else None
                    
                    next_coord_logits = self.transformer_decoder(
                        embedded_seq, 
                        memory,
                        tgt_mask=mask
                    )
                
                # Get the prediction for the next position only
                next_coord = next_coord_logits[:, -1:, :]
                
                # Add some controlled randomness with temperature
                if temperature > 0:
                    # Add Gaussian noise scaled by temperature
                    noise = torch.randn_like(next_coord) * temperature * 0.01
                    next_coord = next_coord + noise
                
                # Ensure coordinates stay in valid range [0, 1]
                next_coord = torch.clamp(next_coord, 0, 1)
                
                # Add to generated sequence
                generated_points.append(next_coord)
                
                # Update current sequence for next iteration
                current_seq = torch.cat(generated_points, dim=1)
            
            # Concatenate all points
            complete_route = torch.cat(generated_points, dim=1)
        
        return complete_route


class RunningRouteDataset(Dataset):
    """Dataset for running routes and map images"""
    def __init__(self, csv_path, img_dirs, transform=None, verbose=False):
        print(f"Loading dataset from {csv_path}...")
        self.data = pd.read_csv(csv_path)
        self.data = self.data.iloc[0:1000, :]
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
        
        # Normalize coordinate values to [0, 1]
        self.data['route_xy_normalized'] = self.data.apply(
            lambda row: self._normalize_coordinates(row['route_xy'], row['image_dims']), 
            axis=1
        )
        
        print("Dataset preparation complete!")
    
    def _normalize_coordinates(self, coords, img_dims):
        """Normalize coordinates to [0, 1] range based on image dimensions"""
        width, height = eval(img_dims) if isinstance(img_dims, str) else img_dims
        normalized = []
        for x, y in coords:
            norm_x = x / width
            norm_y = y / height
            normalized.append((norm_x, norm_y))
        return normalized
    
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
        
        # Get normalized route coordinates
        route_xy = row['route_xy_normalized']
        route_tensor = torch.tensor(route_xy, dtype=torch.float32)
        
        # Prepare conditions: [distance, duration, start_end_dist, heart_rate]
        conditions = torch.tensor([
            row['distance'],
            row['duration'],
            row['start_end_dist'],
            row['avg_heart_rate']
        ], dtype=torch.float32)
        
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
    
    # Create padding mask
    padding_mask = torch.zeros((len(batch_data), max_len), dtype=torch.bool)
    for i, length in enumerate(seq_lengths):
        padding_mask[i, length:] = True
    
    return {
        'image': images,
        'conditions': conditions,
        'input_seq': padded_inputs,
        'target_seq': padded_targets,
        'seq_lengths': seq_lengths,
        'padding_mask': padding_mask
    }


def train_transformer_model(model, data_loader, num_epochs=50, lr=0.0001, 
                           device='cuda', save_interval=5, save_dir='models'):
    """Train the transformer model"""
    print("Initializing training for transformer model...")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Use MSE loss for coordinate regression
    criterion = nn.MSELoss()
    
    # Create log dictionaries
    losses = {'train': []}
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        
        # Initialize epoch metrics
        epoch_loss = 0.0
        
        # Progress bar for this epoch
        progress_bar = tqdm(
            enumerate(data_loader), 
            total=len(data_loader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=True
        )
        
        for i, batch in progress_bar:
            # Move batch data to device
            images = batch['image'].to(device)
            conditions = batch['conditions'].to(device)
            input_seq = batch['input_seq'].to(device)
            target_seq = batch['target_seq'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predicted_coords = model(images, conditions, input_seq)
            
            # Create mask to ignore padded positions
            mask = ~padding_mask.unsqueeze(-1).expand_as(predicted_coords)
            
            # Apply mask and calculate loss
            loss = criterion(predicted_coords[mask], target_seq[mask])
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
            })
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(data_loader)
        losses['train'].append(avg_epoch_loss)
        
        # Update learning rate based on loss
        scheduler.step(avg_epoch_loss)
        
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
            model_path = f"{save_dir}/transformer_model_{epoch}.pt"
            torch.save(model.state_dict(), model_path)
            
        # Generate sample routes
        if epoch % save_interval == 0 or epoch == num_epochs - 1:
            print("Generating sample routes...")
            model.eval()
            with torch.no_grad():
                # Use a small sample of images from the batch
                num_samples = min(4, images.size(0))
                sample_maps = images[:num_samples]
                sample_conditions = conditions[:num_samples]
                
                # Get start points from the input sequences
                start_points = input_seq[:num_samples, 0:1, :]
                
                # Generate routes
                sample_routes = model.generate_route(
                    sample_maps, 
                    sample_conditions,
                    start_point=start_points
                )
                
                # Visualize and save
                sample_path = f"{save_dir}/transformer_sample_epoch_{epoch}.png"
                visualize_routes(sample_maps, sample_routes, sample_path)
                print(f"Sample routes saved to {sample_path}")
    
    # Calculate total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print(f"{'-'*80}")
    print(f"Training completed in {total_time_str}")
    print(f"Final average loss: {avg_epoch_loss:.4f}")
    
    # Save final model
    final_model_path = f"{save_dir}/transformer_model_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return model, losses


def visualize_routes(maps, routes, save_path=None):
    """Visualize generated routes on maps"""
    fig, axes = plt.subplots(len(routes), 1, figsize=(10, 5 * len(routes)))
    
    if len(routes) == 1:
        axes = [axes]
    
    for i, (map_img, route) in enumerate(zip(maps, routes)):
        # Convert map from tensor to numpy for visualization
        map_np = map_img.cpu().permute(1, 2, 0).numpy()
        # Denormalize if necessary
        map_np = (map_np * 0.5) + 0.5
        map_np = np.clip(map_np, 0, 1)
        
        # Convert route from tensor to normalized coordinates
        route_np = route.cpu().numpy()
        
        # Scale route to image dimensions for visualization
        width, height = map_np.shape[1], map_np.shape[0]
        route_scaled = route_np.copy()
        route_scaled[:, 0] *= width
        route_scaled[:, 1] *= height
        
        # Display the map
        axes[i].imshow(map_np)
        
        # Overlay the route
        axes[i].plot(route_scaled[:, 1], route_scaled[:, 0], 'r-', linewidth=2)
        
        # Highlight start and end points
        axes[i].plot(route_scaled[0, 1], route_scaled[0, 0], 'go', markersize=8)  # Start: green
        axes[i].plot(route_scaled[-1, 1], route_scaled[-1, 0], 'bo', markersize=8)  # End: blue
        
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
    batch_size = 8  # Smaller batch size for transformer
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
    
    # Initialize transformer model
    model = RouteTransformer(
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=6,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)
    
    print(f"Transformer model created successfully")
    print(f"Model architecture:")
    print(f"- Embedding dimension: 128")
    print(f"- Attention heads: 8")
    print(f"- Encoder layers: 3")
    print(f"- Decoder layers: 6")
    
    # Train model
    print(f"{'-'*80}")
    trained_model, losses = train_transformer_model(
        model,
        dataloader, 
        num_epochs=num_epochs, 
        device=device,
        save_dir='models/transformer',
        save_interval=5
    )
    
    # Save final model
    print("Saving final model...")
    torch.save(trained_model.state_dict(), "models/transformer/transformer_model_final.pt")
    
    # Plot losses
    print("Generating loss plot...")
    plt.figure(figsize=(10, 5))
    plt.plot(losses['train'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    loss_plot_path = "models/transformer/training_losses.png"
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")
    
    print(f"{'-'*80}")
    print("Training complete!")

if __name__ == "__main__":
    main()