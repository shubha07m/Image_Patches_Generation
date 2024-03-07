In this exercise I have created a program which include the function inputs_to_dataset. Given an image and label mask,
inputs_to_dataset produces image patches and corresponding mask patches to use as a dataset.

Below an example to show how patches are generated from an image with 3 channels.
![alt text](https://github.com/shubha07m/Image_Patches_Generation/blob/shubh/patches.png)

I have implemented the inputs_to_dataset function in two different ways:

**A. Without using patchify package:**
For this I have only used basic packages like numpy and basic python code to develop the **inputs_to_dataset** function.

**B. Using patchify package:**
For this I have only used mainly patchify and numpy to develop the **inputs_to_dataset_pf** function.

Both the functions are defined in ![dataset_extract.ipynb](https://github.com/shubha07m/Image_Patches_Generation/blob/main/dataset_extract.ipynb) file. In ![example_usage.ipynb](https://github.com/shubha07m/Image_Patches_Generation/blob/main/example_usage.ipynb) file I have called the function on same sample input image and input mask to create patches of both type.

Other than **Python 3.8**, the main aditional package used are **numpy 1.24** and **patchify 0.2**. To create a separate 
virtual environment using the used packages ![**requirements.txt**](https://github.com/shubha07m/Image_Patches_Generation/blob/main/requirements.txt) can be used.
