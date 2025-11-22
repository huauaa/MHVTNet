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
```bibtex
@article{ZHANG2026130467,
    title = {Multi-scale hierarchical voxel-aware transformer network for 3D object detection in open-pit mines},
    journal = {Expert Systems with Applications},
    volume = {300},
    pages = {130467},
    year = {2026},
    issn = {0957-4174},
    doi = {https://doi.org/10.1016/j.eswa.2025.130467},
    url = {https://www.sciencedirect.com/science/article/pii/S0957417425040825},
    author = {Huazhen Zhang and Zhongyu Xie and Fan Zhang and Yuqian Zhao}
}
```
## Acknowledgement
This project is mainly based on the following codebases. Thanks for their great works!
* [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)

