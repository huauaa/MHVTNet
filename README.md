# MHVTNet: Multi-Scale Hierarchical Voxel-Aware Transformer Network
MHVTNet is a 3D object detection framework designed for open-pit mining environments, addressing challenges such as sparse point clouds, weak and small targets, large target scale variations and dust-induced noise.
## Data Preparation
If you want to train on [AutoMine](https://automine.cc/) dataset, the data should be organized in the following KITTI-style structure:
```
data
│── automine
│   │── ImageSets/
│   │── training
│   │   │── label/
│   │   │── velodyne/
│   │   │── ...
│   │── testing
│   │   │── velodyne/
│   │   │── ...
│   │── gt_database/
│   │── automine_dbinfos_train.pkl
│   │── automine_infos_test.pkl
│   │── automine_infos_train.pkl
│   │── automine_infos_val.pkl
│   │── automine_infos_trainval.pkl
```
## Training & Testing
To train and test MHVTNet, you need to import the project files into the corresponding folders of [MMDetection3D](https://github.com/open-mmlab/mmdetection3d). After that, you can follow the standard MMDetection3D workflow for training and testing.  
## Citation 
```
The paper describing this project is currently under preparation.  
Citation information will be provided once the paper is officially published.
```
## Acknowledgement
This project is mainly based on the following codebases. Thanks for their great works!
* [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)

