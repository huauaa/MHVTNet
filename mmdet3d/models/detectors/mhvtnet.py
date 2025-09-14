from typing import Tuple

from torch import Tensor
import torch

from model.registry import MODELS
from model.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStage3DDetector
from typing import Dict, List, Tuple, Union
from ...structures.det3d_data_sample import OptSampleList, SampleList
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@MODELS.register_module()
class MHVTNet(SingleStage3DDetector):

    def __init__(self,
                 voxel_encoder: ConfigType,
                 middle_encoder: ConfigType,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)

    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:

        voxel_dict = batch_inputs_dict['voxels']
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, voxel_dict['coors'],
                                batch_size)
        x = self.backbone(x)
        return x
    
    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:

        self.bbox_head.preds_dict = {'vote_offset': torch.empty((0, 3, 224), device=device), 'seed_points': torch.empty((0, 224, 3), device=device)}
        x = self.extract_feat(batch_inputs_dict)

        points = batch_inputs_dict['points']
        losses = self.bbox_head.loss(points, x, batch_data_samples, **kwargs)
        return losses
    
    def _forward(self,
                 batch_inputs_dict: dict,
                 data_samples: OptSampleList = None,
                 **kwargs) -> Tuple[List[torch.Tensor]]:

        x = self.extract_feat(batch_inputs_dict)
        results = self.bbox_head.forward(x)

        return results
