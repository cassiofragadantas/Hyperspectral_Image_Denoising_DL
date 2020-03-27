This repository contains a Hyperspectral Image denoising algorithm that was proposed in:

Dantas C. F., Cohen J.E. and Gribonval R. 'Hyperspectral Image Denoising using Dictionary Learning'. WHISPERS 2019, Amsterdam, Netherlands. (Available at: https://hal.inria.fr/hal-02175630v1)

The proposed technique combines low-rankness and sparsity (through Dictionary learning).

# Usage example

The main script to be run is 'DL_HSI_denoise.m'.
Suppose your image is placed in the matlab variable 'imnoise' (3D array), then run the following code:

[imout, exec_times] = DL_HSI_denoise(imnoise);

where 'imout' contains the final denoised image and 'exec_times' contains the execution times.

# File list and description

* DL_HSI_denoise.m : Main script. Inputs noisy HSI and outputs its denoised version.

Scripts hierarchy:
DL_HSI_denoise.m -> image_denoise_lr.m -> HO_SuKro_DL_ALS.m -> DictUpdateALS2.m

## core/

* image_denoise_lr.m : Sparsity phase of the denoising approach (uses Dictionary Learning).
* HO_SuKro_DL_ALS.m : Dictionary Learning technique used by default (it learns structured dictionaries which are a sum of Kronecker products).
* DictUpdateALS2.m : Dictionary update step within the HO_SuKro_DL_ALS function.

Accelerating the algorithm:
A way to accelerate the algorithm is choosing a fixed DCT dictionary, by setting 'params.algo_type = 2' in 'DL_HSI_denoise.m'.
This avoids the learning phase and saves about half of the total execution time, by the expense of some slight performance reduction.
Another way to accelerate the denoising process is given in the next subsection.

## misc/

* unfold.m : tensor unfolding used in HO-SuKro dictionary update.

The following files are **not used**, unless the parameter 'single_dictionary' is set to 'true' in DL_HSI_denoise. They correspond an alternative implementation in which we learn a single dictionary to denoise all eigen-images (instead of learning a different dictionary per eigen-image). This allow to accelerate the algorithm while losing some performance.

* HO_SuKro_DL_ALS_various_noise : analogous of HO_SuKro_DL_ALS.m for single-dictionary mode.
* image_denoise_various_noise : analogous of image_denoise_lr.m for single-dictionary mode.
* ksvd_various_noise : analogous of ksvd.m (in ksvdbox) for single-dictionary mode.
* ompdenoise2_various_noise : analogous of ompdenoise2.m (in ksvdbox) for single-dictionary mode.

## toolbox/

The following required toolboxes are to be placed in 'toolbox/' folder (in subfolders named as follows):

* omppbox/ :  download link 'http://www.cs.technion.ac.il/%7Eronrubin/Software/ompbox10.zip'
* ksvdbox/ : download link 'http://www.cs.technion.ac.il/%7Eronrubin/Software/ksvdbox13.zip'
* tensorlab_2016-03-28/ : download link 'https://www.tensorlab.net/download.php?t=1585224153&k=259e93b6&e=sXub4M-rQ144P4Q6-vPbPQ'

You may also run the provided 'setup_toolboxes.m' file that will attempt to
install the required toolboxes automatically.

## Relation to HO-SuKro-2.0 repository

This repository is a subset of the HO-SuKro-2.0 repository (which contains all the codes from EUSIPCO 2019 and Whispers 2019 papers using HO-SuKro dictionaries, plus several other tests).
All files here correspond to homonimous files in HO-SuKro-2.0 (revisions from around end of march 2020), the only exception is the DL_HSI_denoise script which is a cleaned version of DL_HSI_denoise_input in HO-SuKro-2.0.
