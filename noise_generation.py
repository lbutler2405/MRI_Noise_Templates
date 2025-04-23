#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import subprocess
import sys

# Dependency check and installation
def install_dependencies():
    """
    Check and install missing dependencies automatically.
    """
    required_packages = ["numpy", "scipy", "nilearn", "nibabel", "tqdm", "matplotlib", "csv", "re", "pandas"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure dependencies are installed
#install_dependencies()

# Import packages (after installation is ensured)
import os
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
## EDITED: ADDED RICE
from scipy.stats import rice
from nilearn.image import mean_img
##
from nilearn.masking import compute_epi_mask
import nibabel as nib
import pickle
import argparse
import matplotlib.pyplot as plt
import random
import gzip
import csv
import re
import pandas as pd
from tqdm import tqdm
import argparse
import traceback 
import json 

def save_data(data, output_file, output_formats, affine=None, header=None):
    """
    Save data in multiple formats, including NIfTI.

    Parameters:
        data (numpy.ndarray): Data to save.
        output_file (str): Base path for output file (without extension).
        output_formats (list): List of formats to save in (e.g., ["nifti", "pickle", "text", "csv"]).
        affine (numpy.ndarray, optional): Affine matrix for NIfTI format (required for saving as NIfTI).
        header (nibabel.Nifti1Header, optional): Header for NIfTI format (required for saving as NIfTI).

    Returns:
        None
    """
    supported_formats = {"nifti", "pickle", "compressed_pickle", "text", "csv"}
    
    for output_format in output_formats:
        if output_format not in supported_formats:
            raise ValueError(f"Unsupported format: {output_format}. Supported formats are {supported_formats}.")

        if output_format == "nifti":
            if affine is None or header is None:
                raise ValueError("Affine and header must be provided to save data in NIfTI format.")
            
            # Save as NIfTI file
            nifti_img = nib.Nifti1Image(data, affine=affine, header=header)
            nib.save(nifti_img, f"{output_file}.nii.gz")
            print(f"Data saved as NIfTI: {output_file}.nii.gz")

        elif output_format == "pickle":
            # Save as a standard pickle file
            with open(f"{output_file}.pkl", "wb") as f:
                pickle.dump(data, f)
            print(f"Data saved as pickle: {output_file}.pkl")

        elif output_format == "compressed_pickle":
            # Save as a compressed pickle file
            with gzip.open(f"{output_file}.pkl.gz", "wb") as f:
                pickle.dump(data, f)
            print(f"Data saved as compressed pickle: {output_file}.pkl.gz")

        elif output_format == "text":
            # Save as a plain text file (flattened)
            np.savetxt(f"{output_file}.txt", data.flatten(), fmt="%.6f")
            print(f"Data saved as text: {output_file}.txt")

        elif output_format == "csv":
            # Save as a CSV file (flattened)
            with open(f"{output_file}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(data.flatten())
            print(f"Data saved as CSV: {output_file}.csv")

# Function to infer file type
def infer_file_type(template_path):
    """
    Infer the file type of the input template based on its extension.

    Parameters:
        template_path (str): Path to the template file.

    Returns:
        str: File type (e.g., "pickle", "compressed_pickle", "text", "csv").

    Raises:
        ValueError: If the file extension is unsupported or the path is invalid.
    """
    if not template_path:
        raise ValueError("template_path cannot be empty.")

    # Mapping of file extensions to types
    file_types = {
        ".pkl": "pickle",
        ".pkl.gz": "compressed_pickle",
        ".txt": "text",
        ".csv": "csv",
        ".nii.gz" : "nifti",
    }

    # Check the file extension
    for ext, file_type in file_types.items():
        if template_path.endswith(ext):
            return file_type

    # If no match, raise an error
    raise ValueError(f"Unsupported file extension for template file: {template_path}")

def plot_and_save_histogram(data, title, save_path, file_format="png", include_std=False):
    """
    Plot and save a histogram of data.

    Parameters:
        data (numpy.ndarray): Data for which the histogram is generated.
        title (str): Title of the histogram.
        save_path (str): Path to save the histogram (without extension).
        file_format (str): File format to save the histogram (default: "png").
        include_std (bool): Whether to include standard deviation bars.

    Returns:
        None
    """
    # Validate inputs
    if not isinstance(data, np.ndarray):
        raise ValueError("Input 'data' must be a numpy array.")
    if not data.size:
        raise ValueError("Input 'data' cannot be empty.")
    if not isinstance(save_path, str) or not save_path:
        raise ValueError("Invalid 'save_path'. Ensure it is a non-empty string.")
    if file_format not in ["png", "jpeg", "tiff", "pdf"]:
        raise ValueError(f"Unsupported file format: {file_format}. Choose from 'png', 'jpeg', 'tiff', 'pdf'.")

    # Ensure directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create the plot
    plt.figure(figsize=(10, 6))
    try:
        if include_std:
            counts, bins = np.histogram(data, bins=100)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            std = np.std(data)
            plt.errorbar(bin_centers, counts, yerr=std, fmt='o', label='Standard Deviation')
            plt.legend()
        else:
            plt.hist(data, bins=100, alpha=0.7)
        plt.title(title)
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")

        # Save the plot
        full_path = f"{save_path}.{file_format}"
        plt.savefig(full_path, dpi=300)
        print(f"Histogram saved as: {full_path}")
    except Exception as e:
        raise IOError(f"Error saving histogram to {save_path}.{file_format}: {e}")
    finally:
        plt.close()

# Generate Noise Template
def generate_noise_templates(
    nifti_file=None,
    output_dir="noise_templates",
    template_shape=(120, 120, 60, 393),
    num_templates=10,
    intensity_filter=1.0,
    output_formats=["pickle", "compressed_pickle", "csv", "text"],
    template_base_name="Noise_Template",
    random_seed=None,
    use_mask="none",  # Choices: "none", "compute", "user"
    mask_file=None,  # User-provided mask file
    #rician_mode=0.0,  # Default Rician noise mode if no mask is used
    #rician_std=1.0,  # Default Rician noise standard deviation if no mask is used
    verbose=True,
    gen_mode="fit",
    rician_params=None
):
    """
    Generate noise templates with optional mask filtering and intensity adjustment.

    Parameters:
        nifti_file (str, optional): Path to NIfTI file for mask processing.
        output_dir (str): Directory to save generated templates.
        template_shape (tuple): Shape of the noise templates.
        num_templates (int): Number of templates to generate.
        intensity_filter (float): Minimum intensity for filtering.
        output_formats (list): Formats to save the templates.
        template_base_name (str): Base name for the templates.
        random_seed (int, optional): Seed for reproducibility.
        use_mask (str): "none" (no mask), "compute" (auto-generated), "user" (use user-provided mask).
        mask_file (str, optional): Path to user-defined mask (required if use_mask="user").
        rician_mode (float): Default mode for Rician noise if no mask is used.
        rician_std (float): Default standard deviation for Rician noise if no mask is used.
        verbose (bool): If True, print detailed logs.
        gen_mode (str): The method by which the noise is generated. By default, noise is generated by fitting the voxel intensities to a Rician distribution and sampling from this distribution
        rician_params (str): The .json files containing the rician parameters

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    if random_seed is not None:
        np.random.seed(random_seed)

    # Step 1: Masking out the brain to extract only noise
    if use_mask == "compute" and nifti_file:    # used when the algorithm automatically computes a mask
        if verbose:
            print("Automatically computing a brain mask...")
        fmri_img = nib.load(nifti_file)
        brain_mask = compute_epi_mask(fmri_img).get_fdata().astype(bool)
        dilated_mask = binary_dilation(brain_mask, iterations=60)
        eroded_mask = binary_erosion(dilated_mask, iterations=3)
        noise_region_mask = ~eroded_mask
        if verbose:
            print("Mask computed. Applying the mask...")
        fmri_data = fmri_img.get_fdata()
        noise_region_mask = noise_region_mask[..., np.newaxis]
        filtered_values = fmri_data * noise_region_mask

        # Flattening the array
        filtered_values = filtered_values.ravel()
        # Removing 0 or negative values
        filtered_values = filtered_values[filtered_values > 0]
        # Removing values which are very large caused by brain/skull spillover
        mean_intensity = np.mean(filtered_values)   # mean intensity
        std_intensity = np.std(filtered_values) # standard deviation
        threshold = mean_intensity + 0.5*std_intensity  # threshold to remove large values
        filtered_values = filtered_values[filtered_values <= threshold] # apply the threshold
        #filtered_values = filtered_values[filtered_values >= intensity_filter]

    elif use_mask == "user" and mask_file:  # used when the user specifies a mask
        if verbose:
            print(f"Using user-provided mask: {mask_file}")
        mask_img = nib.load(mask_file)
        user_mask = mask_img.get_fdata().astype(bool)
        fmri_img = nib.load(nifti_file)
        fmri_data = fmri_img.get_fdata()
        filtered_values = fmri_data[user_mask]
        filtered_values = filtered_values[filtered_values >= intensity_filter]

    elif use_mask == "none":  
        if verbose:
            print("No mask applied. Using default Rician noise parameters.")
        filtered_values = None
    else:
        raise ValueError("Invalid use_mask option. Use 'none', 'compute', or 'user'.")
    
    # Compute Rician parameters if filtered values exist
    if filtered_values is not None and filtered_values.size > 0:
        
        if gen_mode == "fit":    
            ### EDITED PART: fitting a rician distribution to the filtered noise values
            # Shape param b, loc, scale: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit
            if verbose:
                print("Fitting the data to a Rician distribution...")
            
            # If the user provided the rician _parameters json, load that instead of doing the fit
            if rician_params is not None:
                # Using the given params
                print("Rician parameters provided by the user")
                with open(rician_params, 'r') as file:
                    rician_params_json = json.load(file)
                    rician_shape_param = rician_params_json["shape_param"]
                    rician_loc = rician_params_json["location_param"]
                    rician_scale = rician_params_json["scale_param"]
            
            else:
                rician_shape_param, rician_loc, rician_scale = rice.fit(filtered_values)
            
            if verbose:
                print(fr"Rician distribution parameters: Shape parameter={rician_shape_param:.4f}, Scale parameter={rician_scale:.4f}, Location parameter={rician_loc:.4f}")
                # Saving the Rician parameters as a json
                out_ric = os.path.join(output_dir, "rician_params.json")
                rician_params = {
                    "shape_param": rician_shape_param,
                    "location_param": rician_loc,
                    "scale_param": rician_scale
                }
                with open(out_ric, 'w') as file:
                    json.dump(rician_params, file)

        elif gen_mode == "mode":
            rician_mode = np.mean(filtered_values)
            rician_std = np.std(filtered_values)
            
        else:
            raise ValueError("gen_mode option. Use 'fit', or 'mode'.")
                    
    elif use_mask != "none":
        raise ValueError("No noise values meet the intensity filter. Check your input mask or filter.")
    
    # Step 2: Generate noise templates
    for iteration in range(1, num_templates + 1):
        if verbose:
            print(f"Generating template {iteration}...")
        #np.random.seed(iteration)
        noise_template = np.empty(template_shape, dtype=np.float32)
        for vol_idx in tqdm(range(template_shape[3]), desc=f"Template {iteration}"):
            
            if gen_mode == "fit":
                ### EDITED PART: Using the Rician distribution parameters calculated from the fit... sampling the distribution
                noise = rice.rvs(rician_shape_param, loc=rician_loc, scale=rician_scale, size=template_shape[:3])
                noise_template[..., vol_idx] = noise
                ###
            if gen_mode == "mode":
                noise = np.random.normal(rician_mode, rician_std, size=template_shape[:3])
                noise_template[..., vol_idx] = np.sqrt(noise**2 + np.random.normal(0, rician_std, size=noise.shape)**2)
        
        # Step 3: Save the template in desired formats
        template_name = f"{template_base_name}_{iteration}"
        output_file = os.path.join(output_dir, template_name)
        
        for output_format in output_formats:
            if output_format == "pickle":
                with open(f"{output_file}.pkl", "wb") as f:
                    pickle.dump(noise_template, f)
            elif output_format == "compressed_pickle":
                with gzip.open(f"{output_file}.pkl.gz", "wb") as f:
                    pickle.dump(noise_template, f)
            elif output_format == "csv":
                np.savetxt(f"{output_file}.csv", noise_template.flatten(), delimiter=",")
            elif output_format == "text":
                np.savetxt(f"{output_file}.txt", noise_template.flatten())
        if verbose:
            print(f"Saved template {iteration} in formats: {output_formats}")

    print("Noise templates successfully generated.")

def natural_sort_key(s):
    """Extract numbers from strings for natural sorting."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
    
#Convert to NIFTIs
def convert_templates_to_nifti(
    template_dir, nifti_file, output_dir, file_extension=".pkl", 
    which_mode="all", specific_templates=None, random_count=None
):
    """
    Convert templates to NIfTI format.

    Parameters:
        template_dir (str): Directory containing templates to convert.
        nifti_file (str): Reference NIfTI file.
        output_dir (str): Directory to save converted NIfTI files.
        file_extension (str): Extension of templates (e.g., ".pkl", ".pkl.gz").
        which_mode (str): Conversion mode - "all", "random", or "specific".
        specific_templates (list): List of specific templates to convert (used if which_mode="specific").
        random_count (int): Number of random templates to convert (used if which_mode="random").

    Returns:
        None
    """
    # Validate input paths
    if not os.path.isdir(template_dir):
        raise FileNotFoundError(f"Template directory not found: {template_dir}")
    if not os.path.isfile(nifti_file):
        raise FileNotFoundError(f"Reference NIfTI file not found: {nifti_file}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Load NIfTI header and affine
    print(f"Loading NIfTI file for header and affine: {nifti_file}")
    fmri_img = nib.load(nifti_file)
    affine = fmri_img.affine
    header = fmri_img.header
    
    #_______________________
    # EDITED PART 1
    # Getting the dimensions of the fMRI data
    fmri_data = fmri_img.get_fdata()
    fmri_data_shape = fmri_data.shape
    #_______________________

    # Collect all templates in the directory
    all_templates = sorted(
        [f for f in os.listdir(template_dir) if f.endswith(file_extension)],
        key=natural_sort_key
    )
    if not all_templates:
        raise FileNotFoundError(f"No files with extension {file_extension} found in {template_dir}")

    # Determine templates to process based on which_mode
    if which_mode == "all":
        templates = all_templates
        print(f"Processing all templates: {templates}")
    elif which_mode == "random":
        if random_count is None or random_count <= 0:
            raise ValueError("You must specify a valid --random_count for random mode.")
        if len(all_templates) < random_count:
            raise ValueError(f"Requested {random_count} random templates, but only {len(all_templates)} available.")
        templates = random.sample(all_templates, random_count)
        print(f"Randomly selected {random_count} templates: {templates}")
    elif which_mode == "specific":
        if not specific_templates:
            raise ValueError("No specific templates provided for conversion_mode='specific'.")
        missing_templates = [t for t in specific_templates if t not in all_templates]
        if missing_templates:
            raise FileNotFoundError(
                f"The following specific templates were not found: {missing_templates}. Available templates: {all_templates}"
            )
        templates = specific_templates
        print(f"Processing specific templates: {templates}")
    else:
        raise ValueError("Invalid which_mode. Choose from 'all', 'random', or 'specific'.")

    # Process each template
    for template_name in templates:
        print(f"Converting template {template_name}...")
        template_path = os.path.join(template_dir, template_name)

        # Infer file type and load data
        file_type = infer_file_type(template_path)
        if file_type in ["pickle", "compressed_pickle"]:
            with (gzip.open if file_type == "compressed_pickle" else open)(template_path, "rb") as f:
                noise_template = pickle.load(f)
        elif file_type == "csv":
            noise_template = np.loadtxt(template_path, delimiter=",")
        elif file_type == "text":
            noise_template = np.loadtxt(template_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Validate data shape
        if noise_template.ndim != 4:
            raise ValueError(f"Template {template_name} does not have a valid 4D shape.")

        # Convert to NIfTI
        #_____________________________________
        # EDITED PART 2
        input_filename = os.path.splitext(os.path.basename(nifti_file))[0]
        nifti_filename = f"TEMPLATE_{input_filename}_{os.path.splitext(template_name)[0]}.nii.gz"
        nifti_filename = os.path.join(output_dir, nifti_filename)
        #_____________________________________
        output_nifti = nib.Nifti1Image(noise_template, affine=affine, header=header)
        
        #________________________________
        # EDITED PART 3
        # Truncating the template to match the dimensions of the input image
        output_nifti_array = output_nifti.get_fdata()
        output_nifti_array_shape = output_nifti_array.shape
        
        # Check whether the template is larger than or equal to the nifti image
        if np.all(np.array(output_nifti_array_shape) >= np.array(fmri_data_shape)):
            output_nifti_array = output_nifti_array[:fmri_data_shape[0], :fmri_data_shape[1], :fmri_data_shape[2], :fmri_data_shape[3]]
            output_nifti = nib.Nifti1Image(output_nifti_array, affine=affine, header=header)
        # If the template is smaller than the input image, the template is not able to be used
        else:
            raise ValueError(f"Input nifti image has larger dimensions {fmri_data_shape} than the template {output_nifti_array_shape}")
        #________________________________
        
        nib.save(output_nifti, nifti_filename)
        print(f"Template {template_name} converted and saved as NIfTI: {nifti_filename}")

    print("All selected templates successfully converted to NIfTI format.")

# Generate Histograms
def generate_histograms(
    data_dir, data_type="template", mask=None, file_extension=".nii.gz", output_dir=None,
    histogram_format="png", plot_option="save_all_plot_mean", compute_average=False,
    which_mode="all", random_count=None, specific_files=None
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import nibabel as nib
    import gzip
    import pickle
    import random

    print("Initializing histogram generation...")

    # Step 1: Ensure valid inputs
    if data_type not in ["template", "brain"]:
        raise ValueError("data_type must be 'template' or 'brain'.")
    if histogram_format not in ["png", "jpeg", "tiff", "pdf"]:
        raise ValueError(f"Invalid histogram_format: {histogram_format}.")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Step 2: Collect files based on mode
    print("Collecting files for histogram generation...")
    all_files = [f for f in os.listdir(data_dir) if f.endswith(file_extension)]
    if not all_files:
        raise FileNotFoundError(f"No files with extension {file_extension} found in {data_dir}.")

    if which_mode == "all":
        files = all_files
        print("Mode: Generating histograms for all files.")
    elif which_mode == "random":
        if random_count is None or random_count <= 0:
            raise ValueError("You must specify --random_count for which_mode='random'.")
        if random_count > len(all_files):
            raise ValueError(f"random_count exceeds the number of available files ({len(all_files)}).")
        files = random.sample(all_files, random_count)
        print(f"Mode: Saving {len(files)} random histograms and plotting the mean.")
    elif which_mode == "specific":
        if not specific_files:
            raise ValueError("You must specify --specific_files for which_mode='specific'.")
        files = [f for f in specific_files if f in all_files]
        missing_files = [f for f in specific_files if f not in all_files]
        if missing_files:
            raise FileNotFoundError(f"The following files were not found: {missing_files}")
        print("Mode: Saving specific histograms and plotting the mean.")
    else:
        raise ValueError("Invalid which_mode. Choose from 'all', 'random', or 'specific'.")

    if not files:
        raise FileNotFoundError("No files selected for processing.")
    print(f"Number of files selected: {len(files)}")

    # Step 3: Load data with progress bar
    data_list = []
    for file_name in tqdm(files, desc="Loading data for histograms"):
        file_path = os.path.join(data_dir, file_name)
        try:
            if file_extension in [".nii", ".nii.gz"]:
                img = nib.load(file_path)
                data = img.get_fdata()
                if mask is not None:
                    data = data[mask]
            elif file_extension in [".pkl", ".pkl.gz"]:
                with (gzip.open if file_extension == ".pkl.gz" else open)(file_path, "rb") as f:
                    data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
            data_list.append(data.flatten())
        except Exception as e:
            print(f"Skipping {file_name} due to error: {e}")

    if not data_list:
        raise RuntimeError("No valid data was loaded. Ensure the files are accessible and in a supported format.")

    # Step 4: Save individual histograms (only if not save_mean_only)
    if plot_option != "save_mean_only":
        print("Saving individual histograms...")
        for file_name, data_flattened in tqdm(zip(files, data_list), desc="Saving histograms", total=len(files)):
            histogram_path = os.path.join(output_dir, file_name.replace(file_extension, "_histogram"))
            plt.figure()
            plt.hist(data_flattened, bins=100, color='blue', alpha=0.7)
            plt.title(f"Histogram for {file_name}")
            plt.xlabel("Intensity")
            plt.ylabel("Frequency")
            plt.savefig(f"{histogram_path}.{histogram_format}", dpi=300)
            plt.close()

    # Step 5: Generate and plot the mean histogram
    if compute_average and data_list:
        print("Generating the mean histogram...")
        all_data = np.concatenate(data_list)

        # Calculate histogram bins and counts
        num_bins = 100
        counts, bins = np.histogram(all_data, bins=num_bins)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Calculate standard error of the mean (SEM)
        sem = np.zeros(len(bins) - 1)
        for data in tqdm(data_list, desc="Computing SEM"):
            hist, _ = np.histogram(data, bins=bins)
            sem += (hist - counts / len(data_list)) ** 2
        sem = np.sqrt(sem / len(data_list)) / np.sqrt(len(data_list))

        # Plot mean histogram with vertical error bars
        plt.figure(figsize=(10, 6))
        plt.bar(bin_centers, counts, width=np.diff(bins), alpha=0.7, color="lightblue", edgecolor="blue", label="Mean Histogram")
        plt.errorbar(bin_centers, counts, yerr=sem, fmt="o", color="red", markersize=3, elinewidth=1, capsize=5, label="SEM")
        plt.title("Mean Histogram with Standard Error")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.legend()
        if output_dir:
            avg_histogram_path = os.path.join(output_dir, "mean_histogram")
            plt.savefig(f"{avg_histogram_path}.{histogram_format}", dpi=300)
        plt.show()

    print("Histograms successfully generated and saved.")
    return files

## Compute Correlations
def compute_correlations(
    data_dir,
    analysis_type="global-volume",
    mask=None,
    data_type="template",
    file_extension=".nii.gz",
    which_mode="all",
    random_count=None,
    specific_files=None,
    num_voxel_pairs=1000,
    num_volume_pairs=None,
    output_dir=None,
    plot_option="save_all_plot_mean",
    save_results=False,
    plot_format="png"
):
    """
    Compute correlations for global volumes, voxel-time correlations, and voxel pairs.
    Supports 'all', 'random', and 'specific' file modes. Saves and plots results based on options.
    """
    if plot_format not in ["png", "jpeg", "tiff", "pdf"]:
        raise ValueError("Invalid plot format. Choose from 'png', 'jpeg', 'tiff', or 'pdf'.")

    # Ensure mask is ignored when data_type is "template"
    if data_type == "template":
        mask = None

    # Step 1: Collect files based on which_mode
    all_files = [f for f in os.listdir(data_dir) if f.endswith(file_extension)]
    if which_mode == "all":
        files = all_files
    elif which_mode == "random" and random_count:
        files = random.sample(all_files, min(len(all_files), random_count))
    elif which_mode == "specific" and specific_files:
        files = [f for f in specific_files if f in all_files]
    else:
        raise ValueError("Invalid which_mode. Use 'all', 'random', or 'specific'.")

    if not files:
        raise FileNotFoundError("No files found for analysis.")

    correlation_results = []
    detailed_results = []

    # Neighbouring voxel directions (6-connected: ±x, ±y, ±z)
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    pair_index = 1  # Initialize pair index

    # Step 2: Perform correlation analysis
    for file_name in tqdm(files, desc="Processing Files"):
        print(f"Analyzing file: {file_name}")
        file_path = os.path.join(data_dir, file_name)
        img = nib.load(file_path)
        data = img.get_fdata()

        if analysis_type == "voxel-only":
            # Correlate neighbouring voxels
            voxel_coords = np.argwhere(mask) if mask is not None else np.argwhere(data[..., 0] > 0)
            sampled_voxels = random.sample(list(voxel_coords), len(voxel_coords))
            pairs_count = 0
            for x, y, z in sampled_voxels:
                for dx, dy, dz in directions:
                    if pairs_count >= num_voxel_pairs:
                        break
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (
                        0 <= nx < data.shape[0] and 0 <= ny < data.shape[1] and 0 <= nz < data.shape[2]
                        and (mask is None or mask[nx, ny, nz])  # Check mask only if provided
                    ):
                        corr = np.corrcoef(data[x, y, z, :], data[nx, ny, nz, :])[0, 1]
                        detailed_results.append([pair_index, f"({x}, {y}, {z}) - ({nx}, {ny}, {nz})", corr])
                        correlation_results.append(corr)
                        pairs_count += 1
                        pair_index += 1  # Increment pair index
                if pairs_count >= num_voxel_pairs:
                    break

    # Step 3: Save results
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        results_df = pd.DataFrame(detailed_results, columns=["Pair Index", "Coordinates", "Correlation"])
        results_csv_path = os.path.join(output_dir, f"{analysis_type}_correlation_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Results saved to {results_csv_path}")

        # Step 4: Save and plot histograms
    if correlation_results:
        if plot_option in ["save_all_plot_mean", "save_specific_plot_mean", "save_random_plot_mean"]:
            plt.figure(figsize=(8, 6))
            plt.hist(correlation_results, bins=30, alpha=0.7, color="blue", edgecolor="black")
            plt.title("Correlation Histogram")
            plt.xlabel("Correlation")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"mean_correlation_plot.{plot_format}")
            plt.savefig(plot_path, dpi=300)
            print(f"Saved mean correlation histogram to {plot_path}")
            if "plot_mean" in plot_option:
                plt.show()
            plt.close()

        # Calculate and print mean and standard deviation of correlations
        mean_corr = np.mean(correlation_results)
        std_corr = np.std(correlation_results)
        print(f"Mean Correlation: {mean_corr:.4f}")
        print(f"Standard Deviation of Correlations: {std_corr:.4f}")

    print("Correlation analysis completed successfully.")

# Fill brain with noise
def fill_brain_with_noise(
    template_dir, nifti_file, output_dir, file_extension=".pkl",
    which_mode="all", specific_templates=None, random_count=None,
    output_formats=None, user_mask=None, random_seed=None, verbose=True
):
    """
    Fill a brain mask with noise from specified templates, ensuring that the FoV
    outside the brain mask is filled with random values between 0 and 1.
    """

    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Validate inputs
    if which_mode not in ["all", "specific", "random"]:
        raise ValueError("Invalid which_mode. Choose from 'all', 'specific', or 'random'.")
    if which_mode == "specific" and not specific_templates:
        raise ValueError("specific_templates must be provided when which_mode='specific'.")
    if which_mode == "random" and not random_count:
        raise ValueError("random_count must be provided when which_mode='random'.")
    if not output_formats or len(output_formats) == 0:
        raise ValueError("At least one output format must be specified.")

    print(f"Loading NIfTI file for header, affine, and brain mask: {nifti_file}")
    fmri_img = nib.load(nifti_file)
    affine = fmri_img.affine
    header = fmri_img.header

    # Use or process the user-provided brain mask
    if user_mask is not None:
        if verbose:
            print("Using user-provided brain mask.")
        
        # Check if user_mask is already a numpy array
        if isinstance(user_mask, np.ndarray):
            brain_mask = user_mask
        else:
            # Load and resample the user-provided brain mask if needed
            user_mask_img = nib.load(user_mask)
            brain_mask = user_mask_img.get_fdata().astype(bool)

            if brain_mask.shape != fmri_img.shape[:3]:
                from nibabel.processing import resample_from_to
                print("Resampling brain mask to match fMRI image dimensions...")
                resampled_mask = resample_from_to(user_mask_img, fmri_img)
                brain_mask = resampled_mask.get_fdata().astype(bool)
                if verbose:
                    print(f"Resampled brain_mask shape: {brain_mask.shape}")
    else:
        if verbose:
            print("Computing brain mask automatically...")
        brain_mask_img = compute_epi_mask(fmri_img)
        brain_mask = brain_mask_img.get_fdata().astype(bool)

    if not np.any(brain_mask):
        raise ValueError("Computed or provided brain mask is empty. Check your input NIfTI file.")

    os.makedirs(output_dir, exist_ok=True)

    # Collect all templates in the directory
    all_templates = [f for f in os.listdir(template_dir) if f.endswith(file_extension)]
    if not all_templates:
        raise FileNotFoundError(f"No files with extension {file_extension} found in {template_dir}")

    # Determine templates to process based on which_mode
    if which_mode == "all":
        templates = all_templates
        if verbose:
            print(f"Using all templates: {templates}")
    elif which_mode == "specific":
        templates = [f for f in specific_templates if f in all_templates]
        missing_files = [f for f in specific_templates if f not in all_templates]
        if missing_files:
            raise FileNotFoundError(f"The following templates were not found: {missing_files}")
        if verbose:
            print(f"Using specified templates: {templates}")
    elif which_mode == "random":
        templates = random.sample(all_templates, min(len(all_templates), random_count))
        if verbose:
            print(f"Randomly selected {len(templates)} templates: {templates}")

    filled_output_paths = []

    # Process each template
    for template_name in templates:
        if verbose:
            print(f"Filling brain with noise from template {template_name}...")
        template_path = os.path.join(template_dir, template_name)

        # Load template data
        file_type = infer_file_type(template_path)
        if file_type == "pickle":
            with open(template_path, "rb") as f:
                noise_template = pickle.load(f)
        elif file_type == "compressed_pickle":
            with gzip.open(template_path, "rb") as f:
                noise_template = pickle.load(f)
        elif file_type == "csv":
            noise_template = np.loadtxt(template_path, delimiter=",")
        elif file_type == "text":
            noise_template = np.loadtxt(template_path)
        elif file_type == "nifti":
            noise_template = nib.load(template_path).get_fdata()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        if noise_template.shape[:3] != fmri_img.shape[:3]:
            raise ValueError(f"Template shape {noise_template.shape[:3]} does not match NIfTI file shape {fmri_img.shape[:3]}.")

        filled_data = np.zeros(fmri_img.shape, dtype=np.float32)

        # Fill the brain mask and FoV with noise
        for t in tqdm(range(fmri_img.shape[-1]), desc=f"Filling volumes for {template_name}"):
            noise_volume = noise_template[..., t % noise_template.shape[-1]]

            # Generate random noise for the entire FoV (values between 0 and 1)
            volume_with_noise = np.random.uniform(0, 1, size=fmri_img.shape[:3])

            # Replace brain mask region with noise from the template
            volume_with_noise[brain_mask] = np.random.choice(
                noise_volume.flatten(),
                size=np.sum(brain_mask),
                replace=True
            )

            # Assign this volume to the filled data
            filled_data[..., t] = volume_with_noise

        # Save outputs in the specified formats, passing affine and header to save_data
        for fmt in output_formats:
            filled_name = f"{template_name.replace(file_extension, '')}_filled"
            filled_path = os.path.join(output_dir, filled_name)
            save_data(filled_data, filled_path, [fmt], affine=affine, header=header)
            filled_output_paths.append(filled_path)

    # Save metadata for reproducibility
    metadata = {
        "nifti_file": nifti_file,
        "random_seed": random_seed,
        "selected_templates": templates,
        "output_formats": output_formats,
        "which_mode": which_mode,
        "random_count": random_count,
    }
    metadata_path = os.path.join(output_dir, "fill_brain_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    if verbose:
        print(f"Metadata saved to {metadata_path}")

    print(f"Filled brain outputs saved to {output_dir}.")
    return filled_output_paths

## Main Function
def main():
    """
    Main function to run the complete workflow including noise generation,
    histogram plotting, correlation analysis, and filling the brain with noise.
    """
    parser = argparse.ArgumentParser(description="Noise Analysis and Brain Filling Workflow. Author: Dr. Liam Butler. Contributors: Mr. Kristian Galea")

    # General options
    parser.add_argument("--action", type=str, required=True,
                        choices=["generate", "convert", "fill", "histograms", "correlations"],
                        help="Action to perform: generate, convert, fill, histograms, or correlations.")
    parser.add_argument("--data_dir", type=str, help="Directory containing input data or templates.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--file_extension", type=str, default=".nii.gz",
                        help="File extension for input files (supports .nii, .nii.gz, .pkl, .pkl.gz).")
    parser.add_argument("--random_count", type=int, default=None, help="Number of random files/templates to process.")
    parser.add_argument("--specific_files", nargs="+", default=None, help="List of specific files to process.")
    parser.add_argument("--which_mode", type=str, choices=["all", "specific", "random"], default="all",
                        help="Mode for selecting files/templates: all, specific, or random.")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for reproducibility.")

    # Generate options
    parser.add_argument("--nifti_file", type=str, help="Reference NIfTI file for header, affine, and brain mask.")
    parser.add_argument("--template_shape", nargs="+", type=int, help="Shape of the noise template.")

    ### EDITED PART
    parser.add_argument("--noise_gen_method", type=str, choices=["fit", "mode"], default="fit", help="Generates the templates either by fitting the background to a Rice (fit) or through computing the mode.")
    parser.add_argument("--rician_params", type=str, default=None, help="Path to the .json file containing the Rician params (outputted by --verbose)")    
    ###
    
    parser.add_argument("--num_templates", type=int, default=100, help="Number of noise templates to generate.")
    parser.add_argument("--intensity_filter", type=float, default=1.0, help="Minimum intensity for noise filtering.")
    parser.add_argument("--output_formats", nargs="+", default=["pickle"],
                        help="Output formats for generated noise or filled brain (e.g., 'nifti', 'pickle').")
    parser.add_argument("--template_base_name", type=str, default="template",
                        help="Base name for generated noise templates (default: 'template').")
    parser.add_argument("--use_mask", type=str, choices=["none", "compute", "user"], default="none",
                        help="Use mask filtering: 'none', 'compute', or 'user'.")
    parser.add_argument("--mask_file", type=str, help="Path to user-defined mask file (required if use_mask='user').")
    #parser.add_argument("--rician_mode", type=float, default=0.0,
    #                    help="Default Rician mode for noise generation (used if no mask is applied).")
    #parser.add_argument("--rician_std", type=float, default=1.0,
    #                    help="Default Rician standard deviation for noise generation (used if no mask is applied).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output during execution.")

    # Fill options
    parser.add_argument("--user_mask", type=str, help="Path to a user-defined brain mask.")

    # Histogram options
    parser.add_argument("--data_type", type=str, choices=["template", "brain"], default="template",
                        help="Type of data for histograms (template or brain).")
    parser.add_argument("--plot_option", type=str,
                        choices=["save_all_plot_mean", "save_random_plot_mean", "save_specific_plot_mean", "save_mean_only"],
                        default="save_all_plot_mean", help="Options for saving and plotting histograms.")
    parser.add_argument("--compute_average", action="store_true", help="Compute and plot the average histogram.")
    parser.add_argument("--histogram_format", type=str, choices=["png", "jpeg", "tiff", "pdf"], default="png",
                        help="File format for saved histograms.")

    # Correlation options
    parser.add_argument("--analysis_type", type=str, choices=["global-volume", "voxel-volume", "voxel-only"],
                        default="global-volume", help="Type of correlation analysis.")
    parser.add_argument("--num_voxel_pairs", type=int, default=1000, help="Number of voxel pairs to analyze.")
    parser.add_argument("--num_volume_pairs", type=int, default=None, help="Number of volume pairs to analyze.")
    parser.add_argument("--save_results", action="store_true", help="Save correlation results to a file.")
    parser.add_argument("--plot_format", type=str, choices=["png", "jpeg", "tiff", "pdf"], default="png",
                        help="File format for saving correlation histograms.")

    # Parse arguments
    args = parser.parse_args()

    # Set random seed for reproducibility
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

    # Validate required arguments
    if args.action in ["convert", "fill", "histograms", "correlations"] and not args.data_dir:
        raise ValueError("--data_dir is required for the selected action.")
    if args.action == "generate" and not args.nifti_file:
        raise ValueError("--nifti_file is required for generating noise templates.")
        # Validate required arguments
    if args.action in ["convert", "fill", "histograms", "correlations"] and not args.data_dir:
        raise ValueError("--data_dir is required for the selected action.")
    if args.action == "generate" and not args.nifti_file:
        raise ValueError("--nifti_file is required for generating noise templates.")

    try:
        if args.action == "generate":
            generate_noise_templates(
                nifti_file=args.nifti_file,
                output_dir=args.output_dir,
                template_shape=tuple(args.template_shape),
                num_templates=args.num_templates,
                intensity_filter=args.intensity_filter,
                output_formats=args.output_formats,
                template_base_name=args.template_base_name,
                random_seed=args.random_seed,
                use_mask=args.use_mask,
                mask_file=args.mask_file,
                #rician_mode=args.rician_mode,
                #rician_std=args.rician_std,
                verbose=args.verbose,
                gen_mode=args.noise_gen_method,
                rician_params=args.rician_params
            )

        elif args.action == "convert":
            convert_templates_to_nifti(
                template_dir=args.data_dir,
                nifti_file=args.nifti_file,
                output_dir=args.output_dir,
                file_extension=args.file_extension,
                which_mode=args.which_mode,
                specific_templates=args.specific_files,
                random_count=args.random_count
            )

        elif args.action == "fill":
            user_mask = None
            if args.user_mask:
                print("Loading user-provided brain mask...")
                user_mask_img = nib.load(args.user_mask)

                # Check and adjust affine matrices for compatibility
                fmri_img = nib.load(args.nifti_file)
                user_mask_affine = user_mask_img.affine
                fmri_affine = fmri_img.affine

                # Ensure affines are 4x4
                if user_mask_affine.shape != (4, 4):
                    print("Adjusting user mask affine matrix to 4x4...")
                    user_mask_affine = user_mask_affine[:4, :4]
                if fmri_affine.shape != (4, 4):
                    print("Adjusting NIfTI affine matrix to 4x4...")
                    fmri_affine = fmri_affine[:4, :4]

                # Resample the brain mask to match the NIfTI file
                if user_mask_img.shape[:3] != fmri_img.shape[:3] or not np.allclose(user_mask_affine, fmri_affine):
                    print("Resampling brain mask to match NIfTI file dimensions and affine...")
                    from nibabel.processing import resample_from_to
                    resampled_mask = resample_from_to(user_mask_img, (fmri_img.shape[:3], fmri_affine))
                    user_mask = resampled_mask.get_fdata().astype(bool)
                else:
                    user_mask = user_mask_img.get_fdata().astype(bool)

            fill_brain_with_noise(
                template_dir=args.data_dir,
                nifti_file=args.nifti_file,
                output_dir=args.output_dir,
                file_extension=args.file_extension,
                which_mode=args.which_mode,
                specific_templates=args.specific_files,
                random_count=args.random_count,
                output_formats=args.output_formats,
                user_mask=user_mask,  # Pass the preloaded and processed mask
                random_seed=args.random_seed,
                verbose=args.verbose
            )

        elif args.action == "histograms":
            mask = None
            if args.user_mask and args.data_type == "brain":
                mask = nib.load(args.user_mask).get_fdata().astype(bool)

            files = generate_histograms(
                data_dir=args.data_dir,
                data_type=args.data_type,
                mask=mask,
                file_extension=args.file_extension,
                output_dir=args.output_dir,
                histogram_format=args.histogram_format,
                plot_option=args.plot_option,
                compute_average=args.compute_average,
                which_mode=args.which_mode,
                random_count=args.random_count,
                specific_files=args.specific_files
            )
            if files:
                print("Selected files for histogram generation:")
                for file in files:
                    print(f"- {file}")

        elif args.action == "correlations":
            mask = nib.load(args.user_mask).get_fdata().astype(bool) if args.user_mask else None
            compute_correlations(
                data_dir=args.data_dir,
                analysis_type=args.analysis_type,
                mask=mask,
                data_type=args.data_type,
                file_extension=args.file_extension,
                which_mode=args.which_mode,
                random_count=args.random_count,
                specific_files=args.specific_files,
                num_voxel_pairs=args.num_voxel_pairs,
                num_volume_pairs=args.num_volume_pairs,
                output_dir=args.output_dir,
                plot_option=args.plot_option,
                plot_format=args.plot_format,
                save_results=args.save_results
            )

        print(f"Action '{args.action}' completed successfully.")

    except Exception as e:
        print(f"Error during execution of '{args.action}': {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

