import folium
from selenium import webdriver
import time
from PIL import Image, ImageDraw
import io
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
from functools import partial
import os

def plot_route(latitude, longitude, map_filename="route_map.html", image_filename="route_map.png", plt_route=False, download_image=False):
    """
    Plots a route using latitude and longitude lists on an OpenStreetMap-based interactive map.
    Uses fit_bounds to ensure the entire route is visible and captures an image of the map.
    Places bright colored markers at the corners to accurately determine bounds in the image.

    Args:
        latitude (list): List of latitude coordinates.
        longitude (list): List of longitude coordinates.
        map_filename (str): Output filename for the HTML map.
        image_filename (str): Output filename for the map image.
        plt_route (bool): Whether to plot the route line and markers.
        download_image (bool): Whether to save the map as an image.

    Returns:
        tuple: (folium.Map, numpy.ndarray) The generated interactive map and the bounds as a numpy array.
    """
    if not latitude or not longitude or len(latitude) != len(longitude):
        raise ValueError("Latitude and longitude lists must be non-empty and of the same length.")

    # Create a Folium map centered at the first location
    route_map = folium.Map(location=[latitude[0], longitude[0]], zoom_start=14, tiles="OpenStreetMap")

    # Calculate the bounds with padding
    min_lat = min(latitude)
    max_lat = max(latitude)
    min_lon = min(longitude)
    max_lon = max(longitude)
    
    # Add a small padding to the bounds (around 5%)
    lat_padding = (max_lat - min_lat) * 0.1
    lon_padding = (max_lon - min_lon) * 0.1
    
    # Create bounds with padding
    sw = [min_lat - lat_padding, min_lon - lon_padding]
    ne = [max_lat + lat_padding, max_lon + lon_padding]
    
    # Apply bounds immediately after creating the map
    route_map.fit_bounds([sw, ne])
    
    # Store bounds as numpy array
    bounds_array = np.array([sw, ne])

    # Add route to the map
    route = list(zip(latitude, longitude))

    if plt_route:
        folium.PolyLine(route, color="blue", weight=5, opacity=0.7).add_to(route_map)

        # Add start and end markers
        folium.Marker(route[0], popup="Start", icon=folium.Icon(color="green")).add_to(route_map)
        folium.Marker(route[-1], popup="End", icon=folium.Icon(color="red")).add_to(route_map)
    
    # Add corner markers with distinct bright colors for later detection
    # Bottom-left (southwest)
    folium.CircleMarker(
        location=sw,
        radius=.25,
        fill=True,
        fill_color="#FF00FF",  # Magenta
        color="#FF00FF",
        fill_opacity=1.0,
        popup="SW"
    ).add_to(route_map)
    
    # Top-left (northwest)
    folium.CircleMarker(
        location=[ne[0], sw[1]],
        radius=.25,
        fill=True,
        fill_color="#00FFFF",  # Cyan
        color="#00FFFF",
        fill_opacity=1.0,
        popup="NW"
    ).add_to(route_map)
    
    # Bottom-right (southeast)
    folium.CircleMarker(
        location=[sw[0], ne[1]],
        radius=.25,
        fill=True,
        fill_color="#FFFF00",  # Yellow
        color="#FFFF00",
        fill_opacity=1.0,
        popup="SE"
    ).add_to(route_map)
    
    # Top-right (northeast)
    folium.CircleMarker(
        location=ne,
        radius=.25,
        fill=True,
        fill_color="#FF0000",  # Red
        color="#FF0000",
        fill_opacity=1.0,
        popup="NE"
    ).add_to(route_map)

    # Capture the map as an image
    img_data = route_map._to_png()
    img = Image.open(io.BytesIO(img_data))

    if download_image:
        
        # Save the image image
        img.save(image_filename)

    return img, bounds_array

def find_color_locations(image, target_color):
    # Create a boolean mask where True indicates matching pixels
    mask = np.all(image == target_color, axis=2)
    
    # Get the coordinates of True values in the mask
    locations = np.where(mask)
    
    # Return as list of (x, y) tuples
    return list(zip(locations[0], locations[1]))


def find_markers(img):
    """
    Find the colored corner markers in the image and return their center coordinates.
    
    Args:
        img (PIL.Image): The original map image.
        
    Returns:
        dict: Dictionary of marker centers and original image.
    """
    # Convert to RGB if image is RGBA
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Get image as numpy array
    img_array = np.array(img)
    
    # Define the colors to look for (RGB)
    colors = {
        "magenta": (255, 0, 255),  # SW
        "cyan": (0, 255, 255),     # NW
        "yellow": (255, 255, 0),   # SE
        "red": (255, 0, 0)         # NE
    }
    
    # Dictionary to store the center points of each marker
    marker_centers = {}
    
    # Find each marker
    for color_name, color_rgb in colors.items():
        locs = find_color_locations(img_array, color_rgb)

        # Calculate the center of the marker
        if locs:
            x_coords, y_coords = zip(*locs)
            center = (sum(x_coords) // len(x_coords), sum(y_coords) // len(y_coords))
            marker_centers[color_name] = center
        else:
            print(f"Warning: Marker {color_name} not found.")

    # Check if all four markers were found
    if len(marker_centers) != 4:
        print(f"Warning: Not all markers were found. Found {len(marker_centers)} out of 4.")
    
    return {
        "markers": marker_centers,
        "image": img,
        "image_size": img.size  # Store the original image size
    }

def latlon_to_xy(lat, lon, map_data, geo_bounds):
    """
    Convert latitude/longitude to x,y coordinates on the original image.
    
    Args:
        lat (float): Latitude coordinate.
        lon (float): Longitude coordinate.
        map_data (dict): Dictionary containing marker centers and image size.
        geo_bounds (dict): Dictionary with geographic bounds keyed by marker colors.
        
    Returns:
        tuple: (x, y) coordinates on the original image.
    """
    markers = map_data["markers"]
    
    # Check if we have all required markers
    required_markers = ["magenta", "cyan", "yellow", "red"]
    if not all(marker in markers for marker in required_markers):
        return None
    
    # Get geographic bounds
    min_lat, min_lon = geo_bounds[0]
    max_lat, max_lon = geo_bounds[1]
    
    # Calculate relative position in geographic space (0 to 1)
    rel_x = (lon - min_lon) / (max_lon - min_lon)
    rel_y = 1 - (lat - min_lat) / (max_lat - min_lat)  # Invert Y (image coordinates increase downward)
    
    # Get marker positions
    sw_pos = markers["magenta"]
    nw_pos = markers["cyan"]
    se_pos = markers["yellow"]
    ne_pos = markers["red"]
    
    # Apply bilinear interpolation
    # First interpolate along top and bottom edge
    top_x = nw_pos[0] + rel_x * (ne_pos[0] - nw_pos[0])
    bottom_x = sw_pos[0] + rel_x * (se_pos[0] - sw_pos[0])
    
    # Then interpolate between top and bottom
    x = int(top_x + rel_y * (bottom_x - top_x))
    
    # Similarly for y-coordinate
    left_y = nw_pos[1] + rel_y * (sw_pos[1] - nw_pos[1])
    right_y = ne_pos[1] + rel_y * (se_pos[1] - ne_pos[1])
    
    y = int(left_y + rel_x * (right_y - left_y))
    
    return (x, y)


import math
def haversine_distance(lat1, lon1, lat2, lon2):
    # calculate distance between 2 points on Earth

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    # Earth's radius in miles
    radius = 3958.8  # miles (6371 km)
    
    # Calculate distance
    distance = radius * c
    
    return distance

def get_route_distance(latitudes, longitudes):
    total_distance = 0
    # Calculate distance between consecutive points
    for i in range(len(latitudes) - 1):
        distance = haversine_distance(
            latitudes[i], longitudes[i],
            latitudes[i+1], longitudes[i+1]
        )
        total_distance += distance
        
    return total_distance


def calculate_route_duration(timestamps):
        
    # Calculate duration in seconds
    start_time = timestamps[0]
    end_time = timestamps[-1]
    duration_seconds = end_time - start_time
    
    # Convert to minutes
    duration_minutes = duration_seconds / 60
    
    return duration_minutes

def get_start_end_dist(latitudes, longitudes):
    # Calculate distance between consecutive points
    start_lat = latitudes[0]
    start_lon = longitudes[0]
    end_lat = latitudes[-1]
    end_lon = longitudes[-1]
    
    distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    
    return distance

def preprocess_route(row_tuple):
    idx, row = row_tuple
    
    latitude = row['latitude']
    longitude = row['longitude']

    # Use idx for the filename
    route_map, bounds = plot_route(latitude, longitude, plt_route=False, download_image=True, 
                                 image_filename=f"image_data/map{idx}.png")  


    map_data = find_markers(route_map)


    # Get the cropped image size
    image_dims = (route_map.width, route_map.height)

    route_xy = []
    for lat, lon in zip(latitude, longitude):
        xy = latlon_to_xy(lat, lon, map_data, bounds)
        route_xy.append(xy)

    return bounds, route_xy, image_dims

def parallel_process(df, num_processes=None):
    # If num_processes is None, use all available cores
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Convert DataFrame to list of (idx, row) tuples
    row_tuples = list(df.iterrows())

    folder_name = "image_data"

    # Check if the folder already exists
    if not os.path.exists(folder_name):
        # Create the folder
        os.makedirs(folder_name)
    
    # Create a pool of workers with 'spawn' context for better compatibility
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_processes) as pool:
        # Process rows in parallel with progress bar
        results = list(tqdm(
            pool.imap(preprocess_route, row_tuples),
            total=len(df)
        ))
    
    # Unpack results
    bounds_list, route_xy_list, image_dims_list = zip(*results)
    
    # Assign results back to the DataFrame
    df['bounds'] = bounds_list
    df['route_xy'] = route_xy_list
    df['image_dims'] = image_dims_list
    
    return df


def draw_test_preprocess(df, idx):

    row_tuple = list(df.iterrows())[idx]

    idx, row = row_tuple
    
    latitude = row['latitude']
    longitude = row['longitude']

    # Use idx for the filename
    route_map, bounds = plot_route(latitude, longitude, plt_route=True, download_image=False, 
                                 image_filename=f"image_data/map{idx}.png")  


    map_data = find_markers(route_map)


    route_xy = []
    for lat, lon in zip(latitude, longitude):
        xy = latlon_to_xy(lat, lon, map_data, bounds)
        route_xy.append(xy)


    draw_img = route_map.copy()
    draw = ImageDraw.Draw(draw_img)

    # Draw the route points on the image
    for x, y in route_xy:
        draw.ellipse((y-1, x-1, y+1, x+1), fill="purple", outline="white")


    draw_img.show()