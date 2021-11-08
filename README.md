# MinkLoc3D-SI: Point Cloud Based Large-Scale Place Recognition

### Introduction
The 3D LiDAR place recognition aims to estimate a coarse localization in a previously seen environment based on
a single scan from a rotating 3D LiDAR sensor. The existing solutions to this problem include hand-crafted 
point cloud descriptors (e.g., ScanContext, M2DP, LiDAR IRIS) and deep learning-based solutions (e.g., PointNetVLAD, 
PCAN, LPD-Net, DAGC, MinkLoc3D), which are often only evaluated on accumulated 2D scans from the Oxford RobotCat dataset. 
We introduce MinkLoc3D-SI, a sparse convolution-based solution that utilizes spherical coordinates of 3D points and 
processes the intensity of the 3D LiDAR measurements, improving the performance when a single 3D LiDAR scan is used. 
Our method integrates the improvements typical for hand-crafted descriptors (like ScanContext) with the most 
efficient 3D sparse convolutions (MinkLoc3D). Our experiments show improved results on single scans from 3D LiDARs 
(USyd Campus dataset) and great generalization ability (KITTI dataset). Using intensity information on accumulated 
2D scans (RobotCar Intensity dataset) improves the performance, even though spherical representation doesn’t produce 
a noticeable improvement. As a result, MinkLoc3D-SI is suited for single scans obtained from a 3D LiDAR, 
making it applicable in autonomous vehicles.

![Fig1](images/Fig1.png)

### Citation
Paper details will be uploaded after acceptance.
This work is an extension of Jacek Komorowski's [MinkLoc3D](https://github.com/jac99/MinkLoc3D).


### Environment and Dependencies
Code was tested using Python 3.8 with PyTorch 1.7 and MinkowskiEngine 0.5.0 on Ubuntu 18.04 with CUDA 11.0.

The following Python packages are required:
* PyTorch (version 1.7)
* MinkowskiEngine (version 0.5.0)
* pytorch_metric_learning (version 0.9.94 or above)
* numba
* tensorboard
* pandas
* psutil
* bitarray


Modify the `PYTHONPATH` environment variable to include absolute path to the project root folder: 
```export PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/.../.../MinkLoc3D-SI
```

###Datasets
Preprocessed [University of Sydney Campus](http://its.acfr.usyd.edu.au/datasets/usyd-campus-dataset/) dataset (USyd) 
and [Oxford RobotCar](https://robotcar-dataset.robots.ox.ac.uk/) dataset with intensity channel (IntensityOxford) 
available [here](https://chmura.put.poznan.pl/s/5HxyZefrNLp64fj).
Extract the dataset folders on the same directory as the project code, so that you have three folders there: 1) 
IntensityOxford/ 2) MinkLoc3D-SI/ and 3) USyd/.

The pickle files used for positive/negative examples assignment are compatible with the ones introduced in 
[PointNetVLAD](https://github.com/mikacuy/pointnetvlad) and can be generated using the scripts in generating_queries/ 
folder. The benchmark datasets (Oxford and In-house) introduced in PointNetVLAD can also be used following 
the instructions in PointNetVLAD.

Before the network training or evaluation, run the below code to generate pickles with positive and negative point clouds for each anchor point cloud. 
 
```generate pickles
cd generating_queries/ 

# Generate training tuples for the USyd Dataset
python generate_training_tuples_usyd.py

# Generate evaluation tuples for the USyd Dataset
python generate_test_sets_usyd.py

# Generate training tuples for the IntensityOxford Dataset
python generate_training_tuples_intensityOxford.py

# Generate evaluation tuples for the IntensityOxford Dataset
python generate_test_sets_intensityOxford.py
```

### Training
To train **MinkLoc3D-SI** network, prepare the data as described above.
Edit the configuration file (`config/config_usyd.txt` or `config/config_intensityOxford.txt`):
- `num_points` - number of points in the point cloud. Points are randomly subsampled or zero-padding is applied during loading, if there number of points is too big/small
- `max_distance` - maximum used distance from the sensor
- `dataset_name` - **USyd** / **IntensityOxford** / **Oxford**
- `dataset_folder` - path to the dataset folder
- `batch_size_limit` parameter depending on available GPU memory. In our experiments with 10GB of GPU RAM in the case 
of USyd (23k points) the limit was set to 84, for IntensityOxford (4096 points) the limit was 256.

Edit the model configuration file (`models/minkloc_config.txt`):
- version - **MinkLoc3D** / **MinkLoc3D-I** / **MinkLoc3D-S** / **MinkLoc3D-SI** 
- mink_quantization_size - desired quantization:
  - MinkLoc3D/3D-I: **q<sub>x</sub>,q<sub>y</sub>,q<sub>z</sub>** units: [m, m, m]
  - MinkLoc3D-S/3D-SI **q<sub>r</sub>,q<sub>theta</sub>,q<sub>phi</sub>** units: [m, deg, deg]
   ####IntensityOxford and Oxford coordinates are normalized [-1, 1], so the quantization parameters need to be adjusted accordingly!

To train the network, run:

```train
cd training

# To train the desired model on the USyd Dataset
python train.py --config ../config/config_usyd.txt --model_config ../models/minkloc_config.txt

```

### Evaluation

Pre-trained MinkLoc3D-SI trained on USyd is available in the `weights` folder. To evaluate run the following command:

```eval baseline
cd eval

# To evaluate the model trained on the USyd Dataset
python evaluate.py --config ../config/config_usyd.txt --model_config ../models/minkloc_config.txt --weights ../weights/MinkLoc3D-SI-USyd.pth

```

### License
Our code is released under the MIT License (see LICENSE file for details).

### References
1. J. Komorowski, "MinkLoc3D: Point Cloud Based Large-Scale Place Recognition", Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), (2021)
2. M. A. Uy and G. H. Lee, "PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
