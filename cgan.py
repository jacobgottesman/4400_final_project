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

def is_valid_tuple_list(item):
    if not isinstance(item, list):
        return False
    if len(item) != 500:
        return False
    for t in item:
        if not (isinstance(t, tuple) and len(t) == 2):
            return False
        if not all(isinstance(x, int) for x in t):
            return False
    return True
    
class RunningRouteDataset(Dataset):
    def __init__(self, csv_path, img_dirs, transform=None, verbose=False):
        """
        Dataset for running routes and map images
        
        Args:
            csv_path: Path to CSV file with metadata and route information
            img_dir: Directory containing map images
            transform: Transforms to apply to images
            verbose: Whether to print verbose logs during data loading
        """
        print(f"Loading dataset from {csv_path}...")
        self.data = pd.read_csv(csv_path)
        self.data['route_xy'] = self.data['route_xy'].apply(eval)
        self.data = self.data.iloc[0:500, :]
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
        if self.verbose:
            print(f"Loading image for index {idx}")
        
        img_id = row['index']
        
        if self.verbose:
            print(f"Image ID: {img_id}")
        
        if img_id > 50425:
            img_path = os.path.join(self.img_dirs[1], f"map{img_id}.png")
        else:
            img_path = os.path.join(self.img_dirs[0], f"map{img_id}.png")

        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get route coordinates
        route_xy = np.array(eval(row['route_xy'])) if isinstance(row['route_xy'], str) else row['route_xy']
        route_xy = torch.tensor(route_xy, dtype=torch.float32)
        
        # Get conditional parameters
        distance = torch.tensor([row['distance']], dtype=torch.float32)
        duration = torch.tensor([row['duration']], dtype=torch.float32)
        start_end = torch.tensor([row['start_end_dist']], dtype=torch.float32)
        heart_rate = torch.tensor([row['avg_heart_rate']], dtype=torch.float32)
        
        # Combine conditions into one tensor
        conditions = torch.cat([distance, duration, start_end, heart_rate], dim=0)
        
        return {
            'image': image,
            'route': route_xy,
            'conditions': conditions
        }

class MapImageEncoder(nn.Module):
    def __init__(self):
        super(MapImageEncoder, self).__init__()
        # Input: [B, 3, 1366, 683]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=4)  # [B, 32, 341, 170]
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=3)  # [B, 64, 113, 56]
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)  # [B, 128, 56, 28]
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)  # [B, 256, 28, 14]
        self.bn4 = nn.BatchNorm2d(256)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        # Forward pass through the encoder
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Map Image Encoder
        self.map_encoder = MapImageEncoder()
        
        # Process conditional parameters (distance, duration, start_end, heart_rate)
        self.condition_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2)
        )
        
        # Noise processor
        self.noise_processor = nn.Sequential(
            nn.Linear(128, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        
        # Feature processing and reshaping after combining all inputs
        self.feature_processor = nn.Sequential(
            nn.Linear(28 * 14 * 256 + 128 + 1024, 3072),
            nn.LeakyReLU(0.2),
            nn.Linear(3072, 2000),
            nn.LeakyReLU(0.2),
            nn.Linear(2000, 500 * 2),  # Output 500 x-y coordinates
            nn.Sigmoid()  # Normalize to [0,1] range
        )
        
        # Final reshaper to scale coordinates to image dimensions
        self.final_reshape = nn.Linear(500 * 2, 500 * 2)
        
    def forward(self, map_img, conditions, noise):
        # Encode the map image
        map_features = self.map_encoder(map_img)
        map_features = map_features.view(map_features.size(0), -1)  # Flatten
        
        # Process conditions
        condition_features = self.condition_encoder(conditions)
        
        # Process noise
        noise_features = self.noise_processor(noise)
        
        # Concatenate all features
        combined_features = torch.cat([map_features, condition_features, noise_features], dim=1)
        
        # Process combined features to get route
        route = self.feature_processor(combined_features)
        route = route.view(-1, 500, 2)  # Reshape to [batch_size, 500, 2]
        
        # Scale to image dimensions
        route = self.final_reshape(route.view(-1, 500 * 2)).view(-1, 500, 2)
        
        return route


class RouteEncoder(nn.Module):
    def __init__(self):
        super(RouteEncoder, self).__init__()
        
        # Reshape to [B, 2, 500]
        self.conv1d_1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv1d_3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv1d_4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # Input shape: [batch_size, 500, 2]
        x = x.transpose(1, 2)  # Change to [batch_size, 2, 500]
        
        x = self.leaky_relu(self.conv1d_1(x))
        x = self.leaky_relu(self.conv1d_2(x))
        x = self.leaky_relu(self.conv1d_3(x))
        x = self.leaky_relu(self.conv1d_4(x))
        
        x = self.global_avg_pool(x)  # [batch_size, 512, 1]
        x = x.view(x.size(0), -1)  # [batch_size, 512]
        
        return x


class FeatureFusionModule(nn.Module):
    def __init__(self):
        super(FeatureFusionModule, self).__init__()
        
        # Combine route features (512) and map features (256*29*15)
        self.route_projection = nn.Linear(512, 512)
        self.map_projection = nn.Linear(256 * 28 * 14, 256)
        self.condition_projection = nn.Linear(4, 7)
        
        # Feature fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(512 + 256 + 7, 1792),
            nn.LeakyReLU(0.2),
            nn.Linear(1792, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
    def forward(self, route_features, map_features, conditions):
        # Process route features
        route_features = self.route_projection(route_features)
        
        # Flatten and process map features
        map_features = map_features.view(map_features.size(0), -1)
        map_features = self.map_projection(map_features)
        
        # Process conditions
        condition_features = self.condition_projection(conditions)
        
        # Concatenate all features
        combined = torch.cat([route_features, map_features, condition_features], dim=1)
        
        # Process through fusion layers
        fused_features = self.fusion_layers(combined)
        
        return fused_features


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Route encoder
        self.route_encoder = RouteEncoder()
        
        # Map image encoder (shared with Generator)
        self.map_encoder = MapImageEncoder()
        
        # Feature fusion module
        self.feature_fusion = FeatureFusionModule()
        
        # Output heads
        self.validity_head = nn.Linear(512, 1)
        self.aux_distance_head = nn.Linear(512, 1)
        self.aux_duration_head = nn.Linear(512, 1)
        self.aux_heart_rate_head = nn.Linear(512, 1)
        
    def forward(self, route, map_img, conditions):
        # Encode route
        route_features = self.route_encoder(route)
        
        # Encode map
        map_features = self.map_encoder(map_img)
        
        # Fuse features
        fused_features = self.feature_fusion(route_features, map_features, conditions)
        
        # Get outputs
        validity = torch.sigmoid(self.validity_head(fused_features))
        aux_distance = self.aux_distance_head(fused_features)
        aux_duration = self.aux_duration_head(fused_features)
        aux_heart_rate = self.aux_heart_rate_head(fused_features)
        
        return validity, aux_distance, aux_duration, aux_heart_rate


def weights_init(m):
    """
    Initialize network weights with a normal distribution
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

def train_cgan(data_loader, num_epochs=100, lr=0.0002, beta1=0.5, beta2=0.999, 
               device='cuda', save_interval=10, save_dir='models'):
    """
    Train the CGAN model with improved progress logging
    """
    # Initialize models
    print("Initializing generator and discriminator models...")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Apply weight initialization
    print("Applying weight initialization...")
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Setup optimizers
    print("Setting up optimizers...")
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    auxiliary_loss = nn.MSELoss()
    
    # Create log dictionaries
    losses = {'G': [], 'D': [], 'aux': []}
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    # Calculate total number of batches
    total_batches = len(data_loader)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Initialize epoch metrics
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_aux_loss = 0.0
        
        # Progress bar for this epoch
        progress_bar = tqdm(
            enumerate(data_loader), 
            total=total_batches,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=True
        )
        
        for i, batch in progress_bar:
            # Get batch data
            real_maps = batch['image'].to(device)
            real_routes = batch['route'].to(device)
            conditions = batch['conditions'].to(device)
            
            # Configure input
            batch_size = real_maps.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Generate noise
            noise = torch.randn(batch_size, 128).to(device)
            
            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate fake routes
            gen_routes = generator(real_maps, conditions, noise)
            
            # Discriminator evaluates generated routes
            validity, aux_dist, aux_dur, aux_hr = discriminator(gen_routes, real_maps, conditions)
            
            # Calculate generator losses
            g_loss = adversarial_loss(validity, real_labels)
            
            # Auxiliary losses for conditioning
            aux_d_loss = auxiliary_loss(aux_dist, conditions[:, 0].unsqueeze(1))
            aux_t_loss = auxiliary_loss(aux_dur, conditions[:, 1].unsqueeze(1))
            aux_h_loss = auxiliary_loss(aux_hr, conditions[:, 3].unsqueeze(1))
            
            # Combined loss with auxiliary losses
            aux_g_loss = aux_d_loss + aux_t_loss + aux_h_loss
            g_total_loss = g_loss + aux_g_loss
            
            g_total_loss.backward()
            optimizer_G.step()
            
            # -----------------
            # Train Discriminator
            # -----------------
            optimizer_D.zero_grad()
            
            # Discriminator evaluates real routes
            real_validity, real_aux_dist, real_aux_dur, real_aux_hr = discriminator(real_routes, real_maps, conditions)
            
            # Discriminator evaluates generated routes (detached to avoid training G again)
            fake_validity, fake_aux_dist, fake_aux_dur, fake_aux_hr = discriminator(gen_routes.detach(), real_maps, conditions)
            
            # Calculate discriminator losses
            d_real_loss = adversarial_loss(real_validity, real_labels)
            d_fake_loss = adversarial_loss(fake_validity, fake_labels)
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            # Auxiliary losses for conditioning on real data
            aux_real_d_loss = auxiliary_loss(real_aux_dist, conditions[:, 0].unsqueeze(1))
            aux_real_t_loss = auxiliary_loss(real_aux_dur, conditions[:, 1].unsqueeze(1))
            aux_real_h_loss = auxiliary_loss(real_aux_hr, conditions[:, 3].unsqueeze(1))
            
            # Combined auxiliary loss for real data
            aux_d_real_loss = aux_real_d_loss + aux_real_t_loss + aux_real_h_loss
            
            # Total discriminator loss
            d_total_loss = d_loss + aux_d_real_loss
            
            d_total_loss.backward()
            optimizer_D.step()
            
            # Save losses
            losses['G'].append(g_loss.item())
            losses['D'].append(d_loss.item())
            losses['aux'].append(aux_g_loss.item())
            
            # Update epoch losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_aux_loss += aux_g_loss.item()
            
            # Update progress bar with current batch losses
            progress_bar.set_postfix({
                'G': f"{g_loss.item():.4f}",
                'D': f"{d_loss.item():.4f}",
                'Aux': f"{aux_g_loss.item():.4f}"
            })
        
        # Calculate average epoch losses
        avg_g_loss = epoch_g_loss / total_batches
        avg_d_loss = epoch_d_loss / total_batches
        avg_aux_loss = epoch_aux_loss / total_batches
        
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
        print(f"Avg losses - Generator: {avg_g_loss:.4f}, Discriminator: {avg_d_loss:.4f}, Auxiliary: {avg_aux_loss:.4f}")
        print(f"Estimated time remaining: {est_time_str} (completion around {datetime.datetime.now() + datetime.timedelta(seconds=est_time_remaining):%Y-%m-%d %H:%M:%S})")
        
        # Save models periodically
        if epoch % save_interval == 0 or epoch == num_epochs - 1:
            print(f"Saving model checkpoints for epoch {epoch+1}...")
            torch.save(generator.state_dict(), f"{save_dir}/generator_{epoch}.pt")
            torch.save(discriminator.state_dict(), f"{save_dir}/discriminator_{epoch}.pt")
            
            # Save a sample of generated routes
            print("Generating sample routes...")
            with torch.no_grad():
                num_examples = min(4, batch_size)
                sample_maps = real_maps[:num_examples]  # Take first 4 maps from last batch
                sample_conditions = conditions[:num_examples]
                sample_noise = torch.randn(num_examples, 128).to(device)
                sample_routes = generator(sample_maps, sample_conditions, sample_noise)
                
                # Visualize and save sample routes
                sample_path = f"{save_dir}/sample_epoch_{epoch}.png"
                visualize_routes(sample_maps, sample_routes, sample_path)
                print(f"Sample routes saved to {sample_path}")
    
    # Calculate total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print(f"{'-'*80}")
    print(f"Training completed in {total_time_str}")
    print(f"Final losses - Generator: {avg_g_loss:.4f}, Discriminator: {avg_d_loss:.4f}, Auxiliary: {avg_aux_loss:.4f}")
    
    return generator, discriminator, losses


def visualize_routes(maps, routes, save_path=None):
    """
    Visualize generated routes on maps
    """
    fig, axes = plt.subplots(len(routes), 1, figsize=(10, 5 * len(routes)))
    
    if len(routes) == 1:
        axes = [axes]
    
    for i, (map_img, route) in enumerate(zip(maps, routes)):
        # Convert map from tensor to numpy for visualization
        # map_np = map_img.cpu().permute(1, 2, 0).numpy()
        # map_np = (map_np * 0.5) + 0.5  # Unnormalize if using normalization

        map_np = map_img.cpu().permute(1, 2, 0).numpy()
        print(map_np.shape)
        
        # Convert route from tensor to numpy
        route_np = route.cpu().numpy()
        
        # Display the map
        axes[i].imshow(map_np)
        
        # Overlay the route
        # axes[i].plot(route_np[:, 0] * map_np.shape[1], route_np[:, 1] * map_np.shape[0], 'r-', linewidth=2)
        axes[i].plot(route_np[:, 1]* map_np.shape[0], route_np[:, 0] * map_np.shape[1], 'r-', linewidth=1)
        
        axes[i].set_title(f"Generated Route {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main():
    # Check if GPU is available, else MPS, else CPU
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
    num_epochs = 100
    
    print(f"Configuration:")
    print(f"- Data: {data_path}")
    # print(f"- Images: {img_dir}")
    print(f"- Batch size: {batch_size}")
    print(f"- Epochs: {num_epochs}")
    print(f"{'-'*80}")
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((683, 1366)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create dataset and dataloader
    dataset = RunningRouteDataset(data_path, [img_dir1, img_dir2], transform=transform, verbose=False)
    
    print(f"Creating data loader with {len(dataset)} samples")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Train the model
    print(f"{'-'*80}")
    generator, discriminator, losses = train_cgan(
        dataloader, 
        num_epochs=num_epochs, 
        device=device,
        save_dir='models'
    )
    
    # Save final models
    print("Saving final models...")
    torch.save(generator.state_dict(), "models/generator_final.pt")
    torch.save(discriminator.state_dict(), "models/discriminator_final.pt")
    
    # Plot losses
    print("Generating loss plot...")
    plt.figure(figsize=(10, 5))
    plt.plot(losses['G'], label='Generator Loss')
    plt.plot(losses['D'], label='Discriminator Loss')
    plt.plot(losses['aux'], label='Auxiliary Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    
    loss_plot_path = "models/training_losses.png"
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")
    
    print(f"{'-'*80}")
    print("Training complete!")


if __name__ == "__main__":
    main()