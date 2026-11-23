"""
Visualize Temporal Experiment Data as Voronoi Images
-----------------------------------------------------
For the temporal ablation experiment, visualizes VF predictions at different time horizons
as Voronoi images arranged in a circular pattern. Shows mean and standard deviation
across all sampled predictions for 3 randomly selected test samples.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import json

# -------------------------------------------------------------------------
# Argument Parser
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Visualize temporal predictions as Voronoi images')
parser.add_argument('--model', type=str, default='cross_attn', choices=['full', 'cross_attn', 'temporal'],
                    help='Model variant to visualize')
parser.add_argument('--dataset', type=str, default='bern', choices=['wa', 'bern'],
                    help='Dataset variant')
parser.add_argument('--n_samples', type=int, default=3,
                    help='Number of random test samples to visualize')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for sample selection')
parser.add_argument('--samples_dir', type=str, default='/storage/homefs/mc21s017/samples/',
                    help='Directory containing sample files')
parser.add_argument('--data_path', type=str, default='/storage/homefs/mc21s017/data_vf/',
                    help='Path to data directory')
parser.add_argument('--output_dir', type=str, default='/storage/homefs/mc21s017/samples/',
                    help='Output directory for plots')
parser.add_argument('--image_size', type=int, default=61,
                    help='Voronoi image size')
args = parser.parse_args()

np.random.seed(args.seed)

# -------------------------------------------------------------------------
# Voronoi Image Generation
# -------------------------------------------------------------------------
def generate_voronoi_images_given_image_size(data, xy_coordinates, image_size=(61, 61)):
    """ 
    Generates voronoi images using given values for patching colors 
    and given coordiantes for seed points.
    Here image size is fixed or given by the user.
    Inputs: 
        data: NxL data matrix, N number of samples, L number of dimensions
        xy_coordinates: x- and y- coordinates of the seed points, Lx2  matrix
        image_size: (optional) the output image size
    Outputs:
        vor_images: NxRxC voronoi images, 3D tensor -- (number of samples N, number of rows R, number_of_columns C)
    """

    # Number of seed locations/points
    num_locs = xy_coordinates.shape[0]
    num_obs = data.shape[0]

    # start from 0
    x = np.zeros((num_locs,1))
    y = np.zeros((num_locs,1))
    x[:,0] = xy_coordinates[:,0]
    y[:,0] = xy_coordinates[:,1]
    x = x + int(image_size[0]/2)
    y = y + int(image_size[1]/2)
    voronoi_points = np.column_stack((x,y))

    # A grid of full space points including seed ones
    space_coordinates = np.mgrid[0:image_size[0], 0:image_size[1]]
    x_coord = space_coordinates[0, :].flatten() # columns
    y_coord = space_coordinates[1, :].flatten() # rows
    space_coordinates = np.vstack((x_coord,y_coord)).transpose()

    # Define an image
    img_col_size = image_size[0]
    img_row_size = image_size[1]
    img = np.zeros((img_row_size, img_col_size))

    # Fill in image
    vor_images = np.zeros((num_obs, img_row_size, img_col_size))
    for k in range(num_obs):
        value_vector = data[k,:]
        for img_col_ind, img_row_ind in space_coordinates:
            dist = (voronoi_points[:, 0] - img_col_ind)**2 + (voronoi_points[:,1] - img_row_ind)**2
            idx = np.argmin(dist)
            img[img_row_ind, img_col_ind] = value_vector[idx]

        # Have to flip because of matrix conventions (y axis coordinates increase
        # from down to up but a matrix row indices increase from up to down)
        img = np.flipud(img)
        vor_images[k, :, :] = img

    return vor_images

# -------------------------------------------------------------------------
# Load Coordinates
# -------------------------------------------------------------------------
def load_coordinates(dataset, data_path):
    """Load VF coordinates for the specified dataset."""
    if dataset == 'bern':
        coord_file = os.path.join(data_path, 'p24d2.csv')
        coords_df = pd.read_csv(coord_file)
        xy_coords = coords_df[['x', 'y']].values
    elif dataset == 'wa':
        coord_file = os.path.join(data_path, 'x_y.xlsx')
        coords_df = pd.read_excel(coord_file)
        xy_coords = coords_df[['V1', 'V2']].values
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    print(f"Loaded {xy_coords.shape[0]} VF coordinates from {coord_file}")
    return xy_coords

# -------------------------------------------------------------------------
# Load Temporal Samples
# -------------------------------------------------------------------------
def load_temporal_samples(model, dataset, samples_dir):
    """Load temporal samples and metadata."""
    sample_file = os.path.join(samples_dir, f"samples_per_time_{model}_{dataset}.npz")
    meta_file = os.path.join(samples_dir, f"samples_per_time_{model}_{dataset}_meta.json")
    
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"Sample file not found: {sample_file}")
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")
    
    # Load samples
    samples_data = np.load(sample_file)
    
    # Load metadata
    with open(meta_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"\nLoaded temporal samples for {model} on {dataset}")
    print(f"Time labels: {metadata['time_labels']}")
    print(f"Number of samples per test: {metadata['num_samples']}")
    
    return samples_data, metadata

# -------------------------------------------------------------------------
# Create Circular Plot Layout
# -------------------------------------------------------------------------
def create_temporal_voronoi_plot(samples_dict, time_labels, xy_coords, 
                                  test_indices, image_size, output_path):
    """
    Create a plot showing mean and std voronoi images for selected samples across time.
    
    Layout: For each of the 3 test samples, create a row showing:
    - Mean VF for each time point (arranged horizontally)
    - Std VF for each time point (arranged horizontally)
    """
    n_test_samples = len(test_indices)
    n_times = len(time_labels)
    
    # Create figure: 2 rows per test sample (mean and std), n_times columns
    fig, axes = plt.subplots(n_test_samples * 2, n_times, 
                             figsize=(3*n_times, 3*n_test_samples*2))
    
    if n_test_samples == 1:
        axes = axes.reshape(2, n_times)
    
    for sample_idx, test_idx in enumerate(test_indices):
        for time_idx, time_label in enumerate(time_labels):
            # Get samples for this time point and test sample
            # Shape: (n_test, n_samples, L)
            samples_time = samples_dict[time_label][test_idx, :, :]  # (150, L)
            
            # Compute mean and std across the 150 samples
            mean_vf = samples_time.mean(axis=0, keepdims=True)  # (1, L)
            std_vf = samples_time.std(axis=0, keepdims=True)    # (1, L)
            
            # Generate voronoi images
            mean_vor = generate_voronoi_images_given_image_size(
                mean_vf, xy_coords, image_size=(image_size, image_size)
            )[0]  # (61, 61)
            
            std_vor = generate_voronoi_images_given_image_size(
                std_vf, xy_coords, image_size=(image_size, image_size)
            )[0]  # (61, 61)
            
            # Plot mean
            ax_mean = axes[sample_idx * 2, time_idx]
            im_mean = ax_mean.imshow(mean_vor, cmap='viridis', aspect='auto')
            ax_mean.axis('off')
            
            # Add circular mask
            circle = Circle((image_size/2, image_size/2), image_size/2, 
                          transform=ax_mean.transData, fill=False, 
                          edgecolor='white', linewidth=2)
            ax_mean.add_patch(circle)
            
            if sample_idx == 0:
                ax_mean.set_title(f'{time_label}', fontsize=12, fontweight='bold')
            if time_idx == 0:
                ax_mean.set_ylabel(f'Test {test_idx}\nMean', fontsize=10, fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im_mean, ax=ax_mean, fraction=0.046, pad=0.04)
            
            # Plot std
            ax_std = axes[sample_idx * 2 + 1, time_idx]
            im_std = ax_std.imshow(std_vor, cmap='plasma', aspect='auto')
            ax_std.axis('off')
            
            # Add circular mask
            circle = Circle((image_size/2, image_size/2), image_size/2, 
                          transform=ax_std.transData, fill=False, 
                          edgecolor='white', linewidth=2)
            ax_std.add_patch(circle)
            
            if time_idx == 0:
                ax_std.set_ylabel(f'Test {test_idx}\nStd', fontsize=10, fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im_std, ax=ax_std, fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Temporal Predictions - {args.model.upper()} - {args.dataset.upper()}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {output_path}")
    plt.close()

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(f"Visualizing Temporal Experiment: {args.model} on {args.dataset}")
    print("=" * 70)
    
    # Load coordinates
    xy_coords = load_coordinates(args.dataset, args.data_path)
    
    # Load temporal samples
    samples_data, metadata = load_temporal_samples(args.model, args.dataset, args.samples_dir)
    time_labels = metadata['time_labels']
    
    # Check sample shape
    first_time = time_labels[0]
    n_test, n_samples, L = samples_data[first_time].shape
    print(f"\nSample shape: {n_test} test samples × {n_samples} samples × {L} VF locations")
    
    # Verify coordinates match
    if xy_coords.shape[0] != L:
        print(f"WARNING: Coordinate count ({xy_coords.shape[0]}) != VF locations ({L})")
        print(f"Using first {min(xy_coords.shape[0], L)} locations")
        if xy_coords.shape[0] > L:
            xy_coords = xy_coords[:L, :]
        else:
            raise ValueError("Not enough coordinates for VF locations")
    
    # Randomly select test samples
    test_indices = np.random.choice(n_test, size=args.n_samples, replace=False)
    test_indices = np.sort(test_indices)
    print(f"\nSelected test samples: {test_indices}")
    
    # Create visualization
    output_path = os.path.join(args.output_dir, 
                               f'temporal_voronoi_{args.model}_{args.dataset}_samples_{args.n_samples}.png')
    
    create_temporal_voronoi_plot(
        samples_data, time_labels, xy_coords, test_indices, 
        args.image_size, output_path
    )
    
    print("=" * 70)
    print("Visualization complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
