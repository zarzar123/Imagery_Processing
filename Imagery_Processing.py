#!/usr/bin/env python
# coding: utf-8

# In[10]:


import spectral
import os

# Path to your .hdr file
hdr_file_path = '/Users/zarrintasneem/Downloads/GLREFL_Cape_Cod_Jun2016_1_at-sensor_refl_L1G/GLREFL_Cape_Cod_Jun2016_1_at-sensor_refl_L1G.hdr'

# Load the hyperspectral data
img = spectral.open_image(hdr_file_path)

# Display basic information about the file
print(img)

# If you want to access the array data:
data = img.load()

# Show the shape of the data (rows, columns, bands)
print(f"Data shape: {data.shape}")


# In[11]:


import matplotlib.pyplot as plt

# Select bands for false color composite (you can choose bands based on wavelength)
# Example: Red=band 30, Green=band 20, Blue=band 10
rgb_bands = [30, 20, 10]
rgb = img.read_bands(rgb_bands)

# Normalize the data for display
rgb = rgb / rgb.max()

# Display the false color composite
plt.imshow(rgb)
plt.title("False Color Composite")
plt.show()


# In[12]:


# Select a pixel location (row, col)
row, col = 100, 100  # Example coordinates

# Get the spectral data for the pixel, making sure it's 1D
spectrum = data[row, col, :].flatten()

# Plot the spectral signature
plt.plot(spectrum)
plt.title(f"Spectral Signature of Pixel ({row}, {col})")
plt.xlabel("Band")
plt.ylabel("Reflectance")
plt.show()


# In[2]:


pip install h5py matplotlib


# In[23]:


# Assuming you are using the spectral library
print(img.metadata)


# In[27]:


import numpy as np

# Define the bands corresponding to Red, Green, and Blue based on wavelength
rgb_bands = [48, 24, 9]  # Red: Band 48 (~635 nm), Green: Band 24 (~524 nm), Blue: Band 9 (~449 nm)

# Read the selected bands (this gives you radiance data)
radiance_data = img.read_bands(rgb_bands)

# Print statistics for each band
for i, band in enumerate(rgb_bands):
    print(f"Band {band} statistics:")
    print(f"  Min: {np.min(radiance_data[:, :, i])}")
    print(f"  Max: {np.max(radiance_data[:, :, i])}")
    print(f"  Mean: {np.mean(radiance_data[:, :, i])}")
    print(f"  Median: {np.median(radiance_data[:, :, i])}")
    print(f"  Std Dev: {np.std(radiance_data[:, :, i])}")
    print()


# In[ ]:





# In[ ]:





# In[ ]:





# In[28]:


import numpy as np
import matplotlib.pyplot as plt

# Define the bands corresponding to Red, Green, and Blue based on wavelength
rgb_bands = [48, 24, 9]  # Red: Band 48 (~635 nm), Green: Band 24 (~524 nm), Blue: Band 9 (~449 nm)

# Read the selected bands (this gives you radiance data)
radiance_data = img.read_bands(rgb_bands)

# Apply a logarithmic scaling to the radiance data for better visualization
reflectance_data = np.zeros_like(radiance_data)
for i in range(3):  # Apply logarithmic scaling
    reflectance_data[:, :, i] = np.log1p(radiance_data[:, :, i])  # log1p avoids log(0) issues

# Normalize and clip the values for display
reflectance_data = reflectance_data / np.max(reflectance_data)
reflectance_data = np.clip(reflectance_data, 0, 1)

# Use np.transpose to adjust image orientation for display
plt.imshow(np.transpose(reflectance_data, (1, 0, 2)))  # (1, 0, 2) to adjust orientation
plt.title("Logarithmic Scaled True Color Composite")
plt.axis('off')  # Hide the axes
plt.show()


# In[33]:


import numpy as np
import matplotlib.pyplot as plt

# Define the bands corresponding to Red, Green, and Blue based on wavelength
rgb_bands = [48, 24, 9]  # Red: Band 48 (~635 nm), Green: Band 24 (~524 nm), Blue: Band 9 (~449 nm)

# Read the selected bands (this gives you radiance data)
radiance_data = img.read_bands(rgb_bands)

# Plot histograms of the raw pixel values for each band
for i, band in enumerate(rgb_bands):
    plt.figure()
    plt.hist(radiance_data[:, :, i].flatten(), bins=50, color=['r', 'g', 'b'][i], alpha=0.7)
    plt.title(f"Pixel Value Distribution for Band {band}")
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()


# In[6]:


import spectral
import os

# Path to your .hdr file
hdr_file_path = '/Users/zarrintasneem/Downloads/GLREFL_Cape_Cod_Jun2016_1_at-sensor_refl_L1G/GLREFL_Cape_Cod_Jun2016_1_at-sensor_refl_L1G.hdr'

# Load the hyperspectral data
img = spectral.open_image(hdr_file_path)

# Display basic information about the file
print(img)

# If you want to access the array data:
data = img.load()

# Show the shape of the data (rows, columns, bands)
print(f"Data shape: {data.shape}")


# In[37]:


import numpy as np

# Define the bands corresponding to Red, Green, and Blue based on wavelength
rgb_bands = [48, 24, 9]  # Red: Band 48 (~635 nm), Green: Band 24 (~524 nm), Blue: Band 9 (~449 nm)

# Read the selected bands (this gives you radiance data)
radiance_data = img.read_bands(rgb_bands)

# Print sample pixel values from the center of the image
rows, cols = radiance_data.shape[0], radiance_data.shape[1]

# Print the pixel values from the center of the image
for i, band in enumerate(rgb_bands):
    print(f"Pixel values from Band {band} (center of the image):")
    print(radiance_data[rows//2 - 5:rows//2 + 5, cols//2 - 5:cols//2 + 5, i])


# In[42]:


import numpy as np
import matplotlib.pyplot as plt

# Define the bands corresponding to Red, Green, and Blue based on wavelength
rgb_bands = [48, 24, 9]  # Red: Band 48 (~635 nm), Green: Band 24 (~524 nm), Blue: Band 9 (~449 nm)

# Read the selected bands (this gives you radiance data)
radiance_data = img.read_bands(rgb_bands)

# Approximate solar irradiance for the selected bands (replace with actual values if known)
solar_irradiance = np.array([1535, 1820, 2000])  # Example values in W/m²/μm for Red, Green, Blue

# Apply radiometric correction (normalize by solar irradiance)
reflectance_data = np.zeros_like(radiance_data)
for i in range(3):
    reflectance_data[:, :, i] = radiance_data[:, :, i] / solar_irradiance[i]

# Apply logarithmic scaling to enhance small values for better visualization
reflectance_data = np.log1p(reflectance_data)

# Apply a more aggressive brightness factor to enhance visibility
brightness_factor = 50  # Increase significantly for visibility
reflectance_data = reflectance_data * brightness_factor

# Normalize and clip the values for display
reflectance_data = reflectance_data / np.max(reflectance_data)
reflectance_data = np.clip(reflectance_data, 0, 1)

# Print max and min values after scaling for debugging purposes
print("Max value after scaling:", np.max(reflectance_data))
print("Min value after scaling:", np.min(reflectance_data))

# Use np.transpose to adjust image orientation for display
plt.imshow(np.transpose(reflectance_data, (1, 0, 2)))  # (1, 0, 2) to adjust orientation
plt.title("Aggressively Brightened Radiometrically Corrected Composite")
plt.axis('off')  # Hide the axes
plt.show()


# In[ ]:





# In[ ]:





# In[23]:


pip install laspy


# In[ ]:





# In[29]:


pip install lazrs laszip


# In[32]:


pip show laszip


# In[2]:


import laspy
import os

# Path to your .laz file in the Downloads folder
laz_file_path = os.path.expanduser('~/Downloads/472000_3995000.laz')

# Function to read the .laz file and print basic information
def read_laz_file(laz_file_path):
    # Open the .laz file (laspy will auto-detect the backend)
    with laspy.open(laz_file_path) as laz_file:
        print(f"Reading {laz_file_path}")
        
        # Access the header to get metadata
        header = laz_file.header
        print("File Metadata:")
        print(f"  Point Format: {header.point_format}")
        print(f"  Number of Points: {header.point_count}")
        print(f"  Bounds: {header.mins} to {header.maxs}")
        
        # Read the points from the .laz file
        points = laz_file.read()
        
        # Print the first 5 points (as an example)
        print("\nFirst 5 Points:")
        print(points[:5])

# Call the function to read the .laz file
read_laz_file(laz_file_path)


# In[3]:


pip install open3d laspy


# In[ ]:





# In[1]:


import laspy
import numpy as np
import open3d as o3d
import os

# Path to your .laz file in the Downloads folder
laz_file_path = os.path.expanduser('~/Downloads/472000_3995000.laz')

# Function to read and visualize a subset of the .laz file
def visualize_laz_file(laz_file_path, sample_size=100000):
    # Open the .laz file and read the points
    with laspy.open(laz_file_path) as laz_file:
        print(f"Reading {laz_file_path}")
        points = laz_file.read().points
    
    # Extract X, Y, Z coordinates
    x = points.X * laz_file.header.scale[0] + laz_file.header.offset[0]
    y = points.Y * laz_file.header.scale[1] + laz_file.header.offset[1]
    z = points.Z * laz_file.header.scale[2] + laz_file.header.offset[2]
    
    # Stack the coordinates into a Nx3 array
    point_cloud_array = np.vstack((x, y, z)).transpose()
    
    # Randomly sample points to reduce memory load (for large point clouds)
    if sample_size < len(point_cloud_array):
        indices = np.random.choice(len(point_cloud_array), sample_size, replace=False)
        point_cloud_array = point_cloud_array[indices]

    # Create an Open3D PointCloud object
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_array)

    # Visualize the point cloud
    print(f"Visualizing point cloud with {len(point_cloud_array)} points...")
    o3d.visualization.draw_geometries([point_cloud_o3d])

# Visualize a subset of 100,000 points (you can adjust this number)
visualize_laz_file(laz_file_path, sample_size=100000)


# In[61]:


print(f"Min value: {np.min(band_data)}, Max value: {np.max(band_data)}")

# Normalize band data for visualization
band_data_normalized = (band_data - np.min(band_data)) / (np.max(band_data) - np.min(band_data))

# Visualize normalized data
plt.imshow(band_data_normalized, cmap='gray')
plt.title('Band 58 (Normalized)')
plt.colorbar(label='Normalized Reflectance')
plt.show()

plt.imshow(band_data, cmap='inferno')
plt.title('Band 58 (Near-Infrared with Inferno Color Map)')
plt.colorbar(label='Reflectance')
plt.show()


# In[ ]:





# In[46]:


import pandas as pd

# Function to extract elevation features from the point cloud
def extract_elevation_features(laz_file_path, sample_size=100000):
    with laspy.open(laz_file_path) as laz_file:
        points = laz_file.read().points
    
    # Extract X, Y, Z coordinates
    x = points.X * laz_file.header.scale[0] + laz_file.header.offset[0]
    y = points.Y * laz_file.header.scale[1] + laz_file.header.offset[1]
    z = points.Z * laz_file.header.scale[2] + laz_file.header.offset[2]
    
    # Stack into Nx3 array
    point_cloud_array = np.vstack((x, y, z)).transpose()
    
    # Randomly sample points
    if sample_size < len(point_cloud_array):
        indices = np.random.choice(len(point_cloud_array), sample_size, replace=False)
        point_cloud_array = point_cloud_array[indices]
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(point_cloud_array, columns=['X', 'Y', 'Z'])
    
    # Extract basic elevation statistics
    elevation_features = {
        'min_elevation': df['Z'].min(),
        'max_elevation': df['Z'].max(),
        'mean_elevation': df['Z'].mean(),
        'std_elevation': df['Z'].std(),
    }
    
    return elevation_features

# Extract elevation features from the point cloud
elevation_features = extract_elevation_features(laz_file_path, sample_size=100000)
print("Extracted Elevation Features:\n", elevation_features)


# In[63]:


import pandas as pd
import numpy as np
import laspy
import os

# Function to extract elevation features from the LiDAR point cloud
def extract_elevation_features(laz_file_path, sample_size=100000):
    with laspy.open(laz_file_path) as laz_file:
        points = laz_file.read().points
    
    # Extract X, Y, Z coordinates
    x = points.X * laz_file.header.scale[0] + laz_file.header.offset[0]
    y = points.Y * laz_file.header.scale[1] + laz_file.header.offset[1]
    z = points.Z * laz_file.header.scale[2] + laz_file.header.offset[2]
    
    # Stack into Nx3 array
    point_cloud_array = np.vstack((x, y, z)).transpose()
    
    # Randomly sample points to reduce data size
    if sample_size < len(point_cloud_array):
        indices = np.random.choice(len(point_cloud_array), sample_size, replace=False)
        point_cloud_array = point_cloud_array[indices]
    
    # Create a DataFrame for easier manipulation
    df_lidar = pd.DataFrame(point_cloud_array, columns=['X', 'Y', 'Z'])
    
    return df_lidar

# Path to the .laz file (replace with your actual file path)
laz_file_path = os.path.expanduser('~/Downloads/472000_3995000.laz')

# Extract LiDAR elevation features
df_lidar = extract_elevation_features(laz_file_path, sample_size=100000)
print("LiDAR data extracted")

# Assuming you have the hyperspectral data already loaded as `band_data` (e.g., from Band 58 or other bands)
# For simplicity, we'll assume it's a 2D array

# Create an empty DataFrame to combine both LiDAR and Hyperspectral data
combined_df = pd.DataFrame()

# Adding LiDAR elevation data
combined_df['Elevation'] = df_lidar['Z']

# Add hyperspectral data (flattened for matching with LiDAR, assuming similar resolution)
combined_df['Hyperspectral_Band_58'] = band_data.flatten()

# You can add more bands or features as needed
# combined_df['NDVI'] = ndvi.flatten()

# Print the combined data
print(combined_df.head())


# In[ ]:




