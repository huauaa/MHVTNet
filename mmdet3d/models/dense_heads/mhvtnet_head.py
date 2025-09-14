import warnings
from typing import List, Tuple

import numpy as np
import torch
from mmdet.models.utils import multi_apply
from mmdet.utils.memory import cast_tensor_type
from mmengine.runner import amp
from torch import Tensor
from torch import nn as nn

from model.models.task_modules import PseudoSampler
from model.models.test_time_augs import merge_aug_bboxes_3d
from model.registry import MODELS, TASK_UTILS
from model.utils.typing_utils import (ConfigType, InstanceList,
                                        OptConfigType, OptInstanceList)
from .base_3d_dense_head import Base3DDenseHead
from .train_mixins import AnchorTrainMixin

from model.models.layers import VoteModule
from model.models.modules.transformer import Transformer,random_voxel_sampling
from model.models.modules.hcvf import HCVF
from model.models.modules.coordatt3d import CoordAtt3D
from mmdet3d.utils.tools import (random_voxel_sampling, divide_feats, 
                                 get_voxel_centers, get_voxel_coords, 
                                 project_voxel_features)
from mmdet3d.utils.networks import create_fc_layers


from model.structures.det3d_data_sample import SampleList
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from model.structures import BaseInstance3DBoxes
from model.structures.bbox_3d import (DepthInstance3DBoxes,
                                        LiDARInstance3DBoxes,)

from typing import Dict, List, Optional, Tuple, Union
try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@MODELS.register_module()
class MHVTNetHead(Base3DDenseHead, AnchorTrainMixin):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 feat_channels: int = 256,
                 use_direction_classifier: bool = True,
                 vote_module_cfg: Optional[dict] = None,#自定义
                 vote_loss: Optional[dict] = None,#自定义
                 anchor_generator: ConfigType = dict(
                     type='Anchor3DRangeGenerator',
                     range=[0, -39.68, -1.78, 69.12, 39.68, -1.78],
                     strides=[2],
                     sizes=[[3.9, 1.6, 1.56]],
                     rotations=[0, 1.57],
                     custom_values=[],
                     reshape_out=False),
                 assigner_per_size: bool = False,
                 assign_per_class: bool = False,
                 diff_rad_by_sin: bool = True,
                 dir_offset: float = -np.pi / 2,
                 dir_limit_offset: int = 0,
                 bbox_coder: ConfigType = dict(type='DeltaXYZWLHRBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0 / 9.0,
                     loss_weight=2.0),
                 loss_dir: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss', loss_weight=0.2),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.diff_rad_by_sin = diff_rad_by_sin
        self.use_direction_classifier = use_direction_classifier
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.assigner_per_size = assigner_per_size
        self.assign_per_class = assign_per_class
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        warnings.warn(
            'dir_offset and dir_limit_offset will be depressed and be '
            'incorporated into box coder in the future')

        # build anchor generator
        self.prior_generator = TASK_UTILS.build(anchor_generator)
        # In 3D detection, the anchor stride is connected with anchor size
        self.num_anchors = self.prior_generator.num_base_anchors
        # build box coder
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.box_code_size = self.bbox_coder.code_size

        # build loss function
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in [
            'mmdet.FocalLoss', 'mmdet.GHMC'
        ]
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_dir = MODELS.build(loss_dir)
        
        self.vote_module = VoteModule(**vote_module_cfg)
        self.preds_dict = {
                        'vote_offset': torch.empty((0, 3, 224), device=device),
                        'seed_points': torch.empty((0, 224, 3), device=device)
                        }
        self.transformer = Transformer(d_model=32, nhead=4, num_decoder_layers=1, dim_feedforward=512, dropout=0.1 )

        self.hcvf = HCVF(32,32)
        self.coordatt3d=CoordAtt3D(inp=32, oup=32)
        self.vote_loss = MODELS.build(vote_loss)
        self.num_candidates = 224
        
        self._init_layers()
        self._init_assigner_sampler()

        if init_cfg is None:
            self.init_cfg = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=dict(
                    type='Normal', name='conv_cls', std=0.01, bias_prob=0.01))

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = TASK_UTILS.build(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                TASK_UTILS.build(res) for res in self.train_cfg.assigner
            ]

    def _init_layers(self):
        """Initialize neural network layers of the head."""   

        self.cls_out_channels = self.num_anchors * self.num_classes
        self.conv_cls = create_fc_layers(self.feat_channels, self.cls_out_channels)
        self.conv_reg = create_fc_layers(self.feat_channels,
                                  self.num_anchors * self.box_code_size)
        if self.use_direction_classifier:
            self.conv_dir_cls = create_fc_layers(self.feat_channels,
                                          self.num_anchors * 2)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
      
        feats_3d = x[0]
        feats_bev = x[1]
        contextual_voxels=self.hcvf(feats_3d[0],feats_3d[1],feats_3d[2])

        
        contextual_voxels = self.coordatt3d(contextual_voxels)
        

        coords,features = divide_feats(feats_bev)

        voxel_centers = get_voxel_centers(voxel_coords=coords,downsample_times=128,
                                          voxel_size=[0.04, 0.04, 0.375],
                                          point_cloud_range=[1, -39.68, -5, 70.12, 39.68, 10],
                                          dim = 2)#(B, N, 2)
        z_zeros = torch.zeros_like(voxel_centers[..., :1], device = features.device)
        voxel_centers = torch.cat([voxel_centers, z_zeros], dim=-1)#(B, N, 3)
        # print(voxel_centers)
                
        vote_points, vote_feats, offset = self.vote_module(voxel_centers,features)
        self.preds_dict['vote_offset']=torch.cat([self.preds_dict['vote_offset'], offset], dim=0)
        self.preds_dict['seed_points']=torch.cat([self.preds_dict['seed_points'], voxel_centers], dim=0)

        vote_coords = get_voxel_coords(vote_points,
                            downsample_times=8, 
                            voxel_size=[0.04, 0.04, 0.375], 
                            point_cloud_range=[1, -39.68, -5, 70.12, 39.68, 10], dim=3)#([B, N, 3(x,y,z)])
        # print(vote_coords)
        vote_feats = vote_feats.permute(0, 2, 1)
        _, sampled_features, relative_coords = random_voxel_sampling(contextual_voxels, vote_coords, M=32, radius=25)
        
        B, N, M, C = sampled_features.shape
        src = sampled_features.reshape(B * N, M, C).permute(1, 0, 2).contiguous()

        tgt = vote_feats.reshape(1, B * N, C) 
        # print(tgt.shape)
        pos_res = relative_coords.reshape(B * N, M, 3).permute(1, 0, 2).unsqueeze(0).contiguous().float()
        output = self.transformer(src, tgt, pos_res).squeeze(0).view(B, N, C)        

        cls_score = self.conv_cls(output)
        bbox_pred = self.conv_reg(output)
        dir_cls_pred = None
        if self.use_direction_classifier:
            dir_cls_pred = self.conv_dir_cls(output)
        return cls_score, bbox_pred, dir_cls_pred

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:

        return multi_apply(self.forward_single, x)


    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_input_metas,
                 rescale=False,
                 **kwargs):
        aug_bboxes = []
        # only support aug_test for one sample
        for x, input_meta in zip(aug_batch_feats, aug_batch_input_metas):
            outs = self.forward(x)
            bbox_list = self.get_results(*outs, [input_meta], rescale=rescale)
            bbox_dict = dict(
                bboxes_3d=bbox_list[0].bboxes_3d,
                scores_3d=bbox_list[0].scores_3d,
                labels_3d=bbox_list[0].labels_3d)
            aug_bboxes.append(bbox_dict)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, aug_batch_input_metas,
                                            self.test_cfg)
        return [merged_bboxes]

    def get_anchors(self,
                    featmap_sizes: List[tuple],
                    input_metas: List[dict],
                    device: str = 'cuda') -> list:
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            input_metas (list[dict]): contain pcd and img's meta info.
            device (str): device of current module.

        Returns:
            list[list[torch.Tensor]]: Anchors of each image, valid flags
                of each image.
        """
        num_imgs = len(input_metas)
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_anchors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        return anchor_list

    def _loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                             dir_cls_pred: Tensor, labels: Tensor,
                             label_weights: Tensor, bbox_targets: Tensor,
                             bbox_weights: Tensor, dir_targets: Tensor,
                             dir_weights: Tensor, num_total_samples: int):
        """Calculate loss of Single-level results.

        Args:
            cls_score (Tensor): Class score in single-level.
            bbox_pred (Tensor): Bbox prediction in single-level.
            dir_cls_pred (Tensor): Predictions of direction class
                in single-level.
            labels (Tensor): Labels of class.
            label_weights (Tensor): Weights of class loss.
            bbox_targets (Tensor): Targets of bbox predictions.
            bbox_weights (Tensor): Weights of bbox loss.
            dir_targets (Tensor): Targets of direction predictions.
            dir_weights (Tensor): Weights of direction loss.
            num_total_samples (int): The number of valid samples.

        Returns:
            tuple[torch.Tensor]: Losses of class, bbox
                and direction, respectively.
        """
        # classification loss
        if num_total_samples is None:
            num_total_samples = int(cls_score.shape[0])
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        assert labels.max().item() <= self.num_classes
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, self.box_code_size)
        bbox_targets = bbox_targets.reshape(-1, self.box_code_size)
        bbox_weights = bbox_weights.reshape(-1, self.box_code_size)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero(
                        as_tuple=False).reshape(-1)
        num_pos = len(pos_inds)

        pos_bbox_pred = bbox_pred[pos_inds]
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_bbox_weights = bbox_weights[pos_inds]

        # dir loss
        if self.use_direction_classifier:
            dir_cls_pred = dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
            dir_targets = dir_targets.reshape(-1)
            dir_weights = dir_weights.reshape(-1)
            pos_dir_cls_pred = dir_cls_pred[pos_inds]
            pos_dir_targets = dir_targets[pos_inds]
            pos_dir_weights = dir_weights[pos_inds]

        if num_pos > 0:
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                pos_bbox_weights = pos_bbox_weights * bbox_weights.new_tensor(
                    code_weight)
            if self.diff_rad_by_sin:
                pos_bbox_pred, pos_bbox_targets = self.add_sin_difference(
                    pos_bbox_pred, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_bbox_pred,
                pos_bbox_targets,
                pos_bbox_weights,
                avg_factor=num_total_samples)

            # direction classification loss
            loss_dir = None
            if self.use_direction_classifier:
                loss_dir = self.loss_dir(
                    pos_dir_cls_pred,
                    pos_dir_targets,
                    pos_dir_weights,
                    avg_factor=num_total_samples)
        else:
            loss_bbox = pos_bbox_pred.sum()
            if self.use_direction_classifier:
                loss_dir = pos_dir_cls_pred.sum()

        return loss_cls, loss_bbox, loss_dir

    @staticmethod
    def add_sin_difference(boxes1: Tensor, boxes2: Tensor) -> tuple:
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th
                dimensions are changed.
        """
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
            boxes2[..., 6:7])
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(boxes2[...,
                                                                         6:7])
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                           dim=-1)
        return boxes1, boxes2

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            dir_cls_preds: List[Tensor],
            batch_gt_instances_3d: InstanceList,
            batch_input_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None,
            points: List[torch.Tensor] = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d``
                and ``labels_3d`` attributes.
            batch_input_metas (list[dict]): Contain pcd and img's meta info.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, list[torch.Tensor]]: Classification, bbox, and
                direction losses of each level.

                - loss_cls (list[torch.Tensor]): Classification losses.
                - loss_bbox (list[torch.Tensor]): Box regression losses.
                - loss_dir (list[torch.Tensor]): Direction classification
                    losses.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list = self.get_anchors(
            featmap_sizes, batch_input_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.anchor_target_3d(
            anchor_list,
            batch_gt_instances_3d,
            batch_input_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            num_classes=self.num_classes,
            label_channels=label_channels,
            sampling=self.sampling)

        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         dir_targets_list, dir_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        
        losses_vote = self.vote_loss_by_feat(
                points,
                self.preds_dict,
                batch_gt_instances_3d)

        # num_total_samples = None
        with amp.autocast(enabled=False):
            losses_cls, losses_bbox, losses_dir = multi_apply(
                self._loss_by_feat_single,
                cast_tensor_type(cls_scores, dst_type=torch.float32),
                cast_tensor_type(bbox_preds, dst_type=torch.float32),
                cast_tensor_type(dir_cls_preds, dst_type=torch.float32),
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                dir_targets_list,
                dir_weights_list,
                num_total_samples=num_total_samples)
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dir=losses_dir, loss_vote=losses_vote)
        #####base#####
        # return dict(
        #     loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dir=losses_dir)
        
    def loss(self, points: List[torch.Tensor], x: Tuple[Tensor], batch_data_samples: SampleList,
             **kwargs) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(x)

        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        
        batch_pts_semantic_mask = []
        batch_pts_instance_mask = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))
            
            batch_pts_semantic_mask.append(
                data_sample.gt_pts_seg.get('pts_semantic_mask', None))
            batch_pts_instance_mask.append(
                data_sample.gt_pts_seg.get('pts_instance_mask', None))

        loss_inputs = outs + (batch_gt_instances_3d, batch_input_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs, points = points)
        
        # preds_dict={'vote_offset':self.vote_offset,'seed_points':self.seed_point}  
        # vote_loss = self.vote_loss_by_feat_single(points, preds_dict, batch_gt_instances_3d)
        # losses.update(vote_loss)
        
        return losses
    
    def loss_and_predict(self,
                         points: List[torch.Tensor],
                         x: Tuple[Tensor],
                         batch_data_samples: SampleList,
                         proposal_cfg: Optional[ConfigDict] = None,
                         **kwargs) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`Det3DDataSample`]): Each item
                contains the meta information of each image and
                corresponding annotations.
            proposal_cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.

        Returns:
            tuple: the return value is a tuple contains:

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - predictions (list[:obj:`InstanceData`]): Detection
                  results of each image after the post process.
        """
        batch_gt_instances = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        outs = self(x)

        loss_inputs = outs + (batch_gt_instances, batch_input_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs, points=points)
        
        # preds_dict={'vote_offset':self.vote_offset,'seed_points':self.seed_point}  
        # vote_loss = self.vote_loss_by_feat(points, preds_dict, batch_gt_instances)
        
        # losses.update(vote_loss)

        predictions = self.predict_by_feat(
            *outs, batch_input_metas=batch_input_metas, cfg=proposal_cfg)
        return losses, predictions

    def vote_loss_by_feat(
            self,
            points: List[torch.Tensor],
            bbox_preds_dict: dict,
            batch_gt_instances_3d: List[InstanceData],
            **kwargs) -> dict:
        """Compute loss.

        Args:
            points (list[torch.Tensor]): Input points.
            bbox_preds_dict (dict): Predictions from forward of vote head.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.
            batch_pts_semantic_mask (list[tensor]): Semantic mask
                of points cloud. Defaults to None. Defaults to None.
            batch_pts_semantic_mask (list[tensor]): Instance mask
                of points cloud. Defaults to None. Defaults to None.
            batch_input_metas (list[dict]): Contain pcd and img's meta info.
            ret_target (bool): Return targets or not.  Defaults to False.

        Returns:
            dict: Losses of 3DSSD.
        """

        targets = self.get_vote_targets(points, bbox_preds_dict,
                                    batch_gt_instances_3d)
        (vote_targets, vote_mask) = targets

        # calculate vote loss
        vote_loss = self.vote_loss(
            bbox_preds_dict['vote_offset'].transpose(1, 2),
            vote_targets,
            weight=vote_mask.unsqueeze(-1))

        # losses = dict(
        #     vote_loss=vote_loss)

        return vote_loss

    def get_vote_targets(
        self,
        points: List[Tensor],
        bbox_preds_dict: dict = None,
        batch_gt_instances_3d: List[InstanceData] = None,
    ) -> Tuple[Tensor]:
        """Generate targets of 3DSSD head.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            bbox_preds_dict (dict): Bounding box predictions of
                vote head.  Defaults to None.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes`` and ``labels``
                attributes.  Defaults to None.
            batch_pts_semantic_mask (list[tensor]): Semantic gt mask for
                point clouds.  Defaults to None.
            batch_pts_instance_mask (list[tensor]): Instance gt mask for
                point clouds. Defaults to None.

        Returns:
            tuple[torch.Tensor]: Targets of 3DSSD head.
        """
        batch_gt_labels_3d = [
            gt_instances_3d.labels_3d
            for gt_instances_3d in batch_gt_instances_3d
        ]
        batch_gt_bboxes_3d = [
            gt_instances_3d.bboxes_3d
            for gt_instances_3d in batch_gt_instances_3d
        ]

        # find empty example
        for index in range(len(batch_gt_labels_3d)):
            if len(batch_gt_labels_3d[index]) == 0:
                fake_box = batch_gt_bboxes_3d[index].tensor.new_zeros(
                    1, batch_gt_bboxes_3d[index].tensor.shape[-1])
                batch_gt_bboxes_3d[index] = batch_gt_bboxes_3d[index].new_box(
                    fake_box)
                batch_gt_labels_3d[index] = batch_gt_labels_3d[
                    index].new_zeros(1)

        seed_points = [
            bbox_preds_dict['seed_points'][i, :self.num_candidates].detach()
            for i in range(len(batch_gt_labels_3d))
        ]

        (vote_targets, vote_mask) = multi_apply(
                self.get_vote_targets_single, points, batch_gt_bboxes_3d,
                batch_gt_labels_3d, seed_points)

        vote_targets = torch.stack(vote_targets)
        vote_mask = torch.stack(vote_mask)

            
        vote_mask = vote_mask / (vote_mask.sum() + 1e-6)

        return (vote_targets, vote_mask)

    def get_vote_targets_single(self,
                            points: Tensor,
                            gt_bboxes_3d: BaseInstance3DBoxes,
                            gt_labels_3d: Tensor,
                            seed_points: Optional[Tensor] = None,
                            **kwargs):
        """Generate targets of ssd3d head for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            pts_semantic_mask (torch.Tensor): Point-wise semantic
                label of each batch.
            pts_instance_mask (torch.Tensor): Point-wise instance
                label of each batch.
            aggregated_points (torch.Tensor): Aggregated points from
                candidate points layer.
            seed_points (torch.Tensor): Seed points of candidate points.

        Returns:
            tuple[torch.Tensor]: Targets of ssd3d head.
        """

        gt_bboxes_3d = gt_bboxes_3d.to(points.device)
        valid_gt = gt_labels_3d != -1
        gt_bboxes_3d = gt_bboxes_3d[valid_gt]
        gt_labels_3d = gt_labels_3d[valid_gt]

        # Generate fake GT for empty scene
        if valid_gt.sum() == 0:
            vote_targets = points.new_zeros(self.num_candidates, 3)
            vote_mask = points.new_zeros(self.num_candidates, dtype=torch.bool)
            return (vote_targets, vote_mask)
        
        # Vote loss targets
        enlarged_gt_bboxes_3d = gt_bboxes_3d.enlarged_box(0.05) #expand_dims_length=0.05
        enlarged_gt_bboxes_3d.tensor[:, 2] -= 0.05
        vote_mask, vote_assignment = self._assign_targets_by_points_inside(
            enlarged_gt_bboxes_3d, seed_points)

        vote_targets = gt_bboxes_3d.gravity_center
        vote_targets = vote_targets[vote_assignment] - seed_points
        vote_mask = vote_mask.max(1)[0] > 0

        return (vote_targets, vote_mask)


    def _assign_targets_by_points_inside(self, bboxes_3d: BaseInstance3DBoxes,
                                            points: Tensor) -> Tuple:
        """Compute assignment by checking whether point is inside bbox.

        Args:
            bboxes_3d (BaseInstance3DBoxes): Instance of bounding boxes.
            points (torch.Tensor): Points of a batch.

        Returns:
            tuple[torch.Tensor]: Flags indicating whether each point is
                inside bbox and the index of box where each point are in.
        """
        if isinstance(bboxes_3d, (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            points_mask = bboxes_3d.points_in_boxes_all(points)
            assignment = points_mask.argmax(dim=-1)
        else:
            raise NotImplementedError('Unsupported bbox type!')

        return points_mask, assignment