from typing import Callable, List, Union

import numpy as np

from model.registry import DATASETS
from model.structures import CameraInstance3DBoxes,LiDARInstance3DBoxes
from .det3d_dataset import Det3DDataset
import sys

@DATASETS.register_module()
class AutomineDataset(Det3DDataset):

    METAINFO = {
        'classes': ('Construction_Vehicles', 'Civilian_Vehicles', 'Pedestrian'),
        'palette': [
            (255, 0, 0),  # red
            (0, 255, 0),  # green
            (0, 0, 255),  # blue
        ]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True),
                 default_cam_key: str = 'CAM2',
                 load_type: str = 'frame_based',
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 pcd_limit_range: List[float] = [1, -39.68, -5, 70.12, 39.68, 10],
                 **kwargs) -> None:

        self.pcd_limit_range = pcd_limit_range
        assert load_type in ('frame_based', 'mv_image_based',
                             'fov_image_based')
        self.load_type = load_type
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            default_cam_key=default_cam_key,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert self.modality is not None
        assert box_type_3d.lower() in ('lidar', 'camera')

    def parse_data_info(self, info: dict) -> dict:

        if self.modality['use_lidar']:
            if 'plane' in info:

                plane = np.array(info['plane'])
                lidar2cam = np.array(
                    info['images']['CAM2']['lidar2cam'], dtype=np.float32)
                reverse = np.linalg.inv(lidar2cam)

                (plane_norm_cam, plane_off_cam) = (plane[:3],
                                                   -plane[:3] * plane[3])
                plane_norm_lidar = \
                    (reverse[:3, :3] @ plane_norm_cam[:, None])[:, 0]
                plane_off_lidar = (
                    reverse[:3, :3] @ plane_off_cam[:, None][:, 0] +
                    reverse[:3, 3])
                plane_lidar = np.zeros_like(plane_norm_lidar, shape=(4, ))
                plane_lidar[:3] = plane_norm_lidar
                plane_lidar[3] = -plane_norm_lidar.T @ plane_off_lidar
            else:
                plane_lidar = None

            info['plane'] = plane_lidar

        if self.load_type == 'fov_image_based' and self.load_eval_anns:
            info['instances'] = info['cam_instances'][self.default_cam_key]

        info = super().parse_data_info(info)

        return info

    def parse_ann_info(self, info: dict) -> dict:

        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

            if self.load_type in ['fov_image_based', 'mv_image_based']:
                ann_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
                ann_info['gt_bboxes_labels'] = np.array(0, dtype=np.int64)
                ann_info['centers_2d'] = np.zeros((0, 2), dtype=np.float32)
                ann_info['depths'] = np.zeros((0), dtype=np.float32)


        ann_info = self._remove_dontcare(ann_info)
        lidar2cam = np.array(info['images']['CAM2']['lidar2cam'])

        gt_bboxes_3d = CameraInstance3DBoxes(
            ann_info['gt_bboxes_3d']).convert_to(self.box_mode_3d,
                                                 np.linalg.inv(lidar2cam))
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info
