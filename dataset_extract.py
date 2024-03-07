#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ------------------------------------------------
# dataset_extract.ipynb
# shubha07m
# 2024-02-05
# dataset_extract file contains "inputs_to_dataset" (uses only numpy) and 
# "inputs_to_dataset_pf" (uses numpy and patchify package) function to
# generate patches of image and masks
# ------------------------------------------------


# ## Functions for custom patches generation from provided image and mask ##

# In[2]:


# Importing the required library #

import numpy as np
from patchify import patchify


# ### Patches generation function without patchify package ###

# In[3]:


def inputs_to_dataset(image, mask, patch_size, stride_length):
    """
    Generates image and mask patches for dataset creation.

    Args:
        image: The input image as a NumPy array.
        mask: The corresponding mask as a NumPy array.
        patch_size: The size of the patches to extract.
        stride_length: The stride length between patches.

    Returns:
        A tuple containing two NumPy arrays:
            - image_patches: A 4D array of image patches.
            - mask_patches: A 3D array of corresponding mask patches.
    """

    # Error message for wrong patch size
    if patch_size >= image.shape[0] or patch_size >= image.shape[1]:
        raise ValueError("Given patch size is greater than or equal to the image height or width")

    # Error message for wrong stride length
    if stride_length >= image.shape[0] or stride_length >= image.shape[1]:
        raise ValueError("Given stride length is greater than or equal to the image height or width")

    # Error message for invalid patch size or stride length
    if patch_size <= 0 or stride_length <= 0:
        raise ValueError("Patch size and stride length must be positive integers.")


    # Error message for mask size greater than image size
    if image.shape[0] < mask.shape[0] or image.shape[1] < mask.shape[1]:
        raise ValueError("Given mask size is greater than image size")


    # Generate image and mask patches
    image_patch_shape = patch_size, patch_size, image.shape[2]  # Shape of each image patch
    mask_patch_shape = patch_size, patch_size  # Shape of each mask patch

    # Initializing empty lists for storing patches #
    filtered_image_patches = []
    filtered_mask_patches = []
    
    # Iterating through image and mask and storing patches in lists
    for i in range(0,image.shape[0]-1,stride_length):
        for j in range(0,image.shape[1]-1,stride_length):
            img_patch = image[i:patch_size+i,j:patch_size+j]
            mask_patch = mask[i:patch_size+i,j:patch_size+j]

            # Filter patches based on condition (exclude those with -1 values)
            if -1 not in img_patch:
                filtered_image_patches.append(img_patch)
                filtered_mask_patches.append(mask_patch)

    # Return image patches as a 4D array and mask patches as a 3D array
    return np.array(filtered_image_patches), np.array(filtered_mask_patches)


# ### Patches generation function with patchify package ###

# In[4]:


def inputs_to_dataset_pf(image, mask, patch_size, stride_length):
    """
    Generates image and mask patches for dataset creation.

    Args:
        image: The input image as a NumPy array.
        mask: The corresponding mask as a NumPy array.
        patch_size: The size of the patches to extract.
        stride_length: The stride length between patches.

    Returns:
        A tuple containing two NumPy arrays:
            - image_patches: A 4D array of image patches.
            - mask_patches: A 3D array of corresponding mask patches.
    """

    # Error message for wrong patch size
    if patch_size >= image.shape[0] or patch_size >= image.shape[1]:
        raise ValueError("Given patch size is greater than or equal to the image height or width")

    # Error message for wrong stride length
    if stride_length >= image.shape[0] or stride_length >= image.shape[1]:
        raise ValueError("Given stride length is greater than or equal to the image height or width")

    # Error message for invalid patch size or stride length
    if patch_size <= 0 or stride_length <= 0:
        raise ValueError("Patch size and stride length must be positive integers.")

    # Error message for mask size greater than image size
    if image.shape[0] < mask.shape[0] or image.shape[1] < mask.shape[1]:
        raise ValueError("Given mask size is greater than image size")

    image_patch_shape = patch_size, patch_size, image.shape[2]  # Shape of each image patch
    mask_patch_shape = patch_size, patch_size  # Shape of each mask patch

    # Generate image and mask patches using the patchify function

    try:
        image_patches = patchify(image, image_patch_shape, step=stride_length)
        mask_patches = patchify(mask, mask_patch_shape, step=stride_length)

    except Exception as e:
        raise ValueError("Error while using patchify function", e) from e

    # Reducing the dimension of generated patches from 6 to 4 #
    image_patches = image_patches.reshape(-1, image_patch_shape[0], image_patch_shape[1], image_patch_shape[2])

    # Reducing the dimension of generated patches from 4 to 3 #
    mask_patches = mask_patches.reshape(-1, mask_patch_shape[0], mask_patch_shape[1])

    # Initializing empty lists for storing patches #
    filtered_image_patches = []
    filtered_mask_patches = []

    # Iterating through image and mask patches and storing in arrays
    for img_patch, mask_patch in zip(image_patches, mask_patches):
        # Filter patches based on condition (exclude those with -1 values)
        if -1 not in img_patch:
            filtered_image_patches.append(img_patch)
            filtered_mask_patches.append(mask_patch)

    # Return image patches as a 4D array and mask patches as a 3D array
    return np.array(filtered_image_patches), np.array(filtered_mask_patches)


# In[5]:


# !jupyter nbconvert --to script dataset_extract.ipynb

