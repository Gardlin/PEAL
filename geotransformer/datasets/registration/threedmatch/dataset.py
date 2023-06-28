import os.path as osp
import pickle
import random
from typing import Dict

import numpy as np
import torch
import torch.utils.data
import os
from geotransformer.utils.pointcloud import (
    random_sample_rotation,
    random_sample_rotation_v2,
    get_transform_from_rotation_translation,
)
from geotransformer.utils.registration import get_correspondences
from PIL import Image, ImageFilter
from torchvision.transforms import transforms
_imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
# from projection import Projection
from visualize import save_ply, draw_registration_result, unproject, adjust_intrinsic,make_open3d_point_cloud

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

from geotransformer.utils.pointcloud import (
    random_sample_rotation,
    random_sample_rotation_v2,
    get_transform_from_rotation_translation,
)
class ThreeDMatchPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1,
        overlap_threshold=None,
        return_corr_indices=False,
        matching_radius=None,
        rotated=False,
        cfg=None,
        iteration=0
    ):
        super(ThreeDMatchPairDataset, self).__init__()
        self.image_path=cfg.train.image_path
        self.geo_train_prior=cfg.train.geo_train_prior
        self.superglue_prior=cfg.train.superglue_prior
        self.using_geo_prior=cfg.train.using_geo_prior
        self.overlap_min_points=cfg.train.overlap_min_points
        self.using_2d_prior=cfg.train.using_2d_prior
        self.window_size=cfg.train.superglue_window_size
        if cfg.test.using_iter_prior and iteration>0:
            print('using iter prior input')
            self.iterative_prior = osp.join(cfg.feature_dir,subset)
            self.geo_prior=self.iterative_prior
        else:
            print('using 3dprior input')
            self.geo_prior=cfg.train.geo_prior
        self.image_size =[120,160]#[480,640]
        self.depth_img_size =[120,160]#[480,640]
        self.transforms = {}
        self.use_augmentation = use_augmentation
        if self.use_augmentation:

            self.transforms["rgb"] = transforms.Compose([
                transforms.Resize(self.image_size, Image.NEAREST),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(_imagenet_stats['mean'], std=_imagenet_stats['std'])
            ])
        else:
            self.transforms["rgb"] = transforms.Compose([
                transforms.Resize(self.image_size, Image.NEAREST),
                transforms.ToTensor()
            ])
        self.transforms["depth"] = transforms.Compose([
            transforms.Resize(self.depth_img_size, Image.NEAREST),
            transforms.ToTensor(),
        ])

        self.dataset_root = dataset_root
        self.metadata_root = osp.join(self.dataset_root, 'metadata')
        self.data_root = osp.join(self.dataset_root, 'data')

        self.subset = subset
        self.point_limit = point_limit
        self.overlap_threshold = overlap_threshold
        self.rotated = rotated

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation

        with open(osp.join(self.metadata_root, f'{subset}.pkl'), 'rb') as f:
            self.metadata_list = pickle.load(f)
            src_list,tgt_list=[],[]
            for data in self.metadata_list:
                if 'pcd0' in data.keys():
                    tgt_list.append(data['pcd0'])
                if 'pcd1' in data.keys():
                    src_list.append(data['pcd1'])
            if self.overlap_threshold is not None:
                self.metadata_list = [x for x in self.metadata_list if x['overlap'] > self.overlap_threshold]
            self.src_scene_id_list, self.src_full_scene_id_list, self.src_seq_id_list, self.src_image_id1_list, self.src_image_id2_list = self.split_info(
                src_list)
            self.tgt_scene_id_list, self.tgt_full_scene_id_list, self.tgt_seq_id_list, self.tgt_image_id1_list, self.tgt_image_id2_list = self.split_info(
                tgt_list)
            if self.overlap_threshold is not None:
                self.metadata_list = [x for x in self.metadata_list if x['overlap'] > self.overlap_threshold]

    def __len__(self):
        return len(self.metadata_list)
    def split_info(self, list, scene_names=None):
        scene_id_list = []
        full_scene_id_list = []
        seq_id_list = []
        image_id1 = []
        image_id2 = []

        for i, fname in enumerate(list):
            phase, scene_id, image_id = fname.split('/')
            # for i in range(len(scene_names)):
            #     if scene_id[:-3] in scene_names[i]:
            #         scene_id=scene_names[i]
            txt_path = image_id[:-4] + '.info.txt'
            with open(os.path.join(self.data_root , phase, scene_id, txt_path), 'r') as f:
                line = f.readline()
                full_scene_id, seq_id, id1, id2 = line.split()
            f.close()
            scene_id_list.append(scene_id)
            full_scene_id_list.append(full_scene_id)
            seq_id_list.append(seq_id)
            image_id1.append(id1)
            image_id2.append(id2)
        return scene_id_list, full_scene_id_list, seq_id_list, image_id1, image_id2
    def _load_point_cloud(self, file_name):
        points = torch.load(osp.join(self.data_root, file_name))
        # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points

    def _augment_point_cloud(self, ref_points, src_points, rotation, translation,aug_src,estimated_rotation,estimated_translation):
        r"""Augment point clouds.

        ref_points = src_points @ rotation.T + translation

        1. Random rotation to one point cloud.
        2. Random noise.
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        if aug_src > 0.5:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)
            estimated_rotation = np.matmul(estimated_rotation, aug_rotation.T)

        else:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            estimated_rotation = np.matmul(aug_rotation, estimated_rotation)
            translation = np.matmul(aug_rotation, translation)
            estimated_translation = np.matmul(aug_rotation, estimated_translation)

        ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.aug_noise
        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise

        return ref_points, src_points, rotation, translation,aug_rotation,estimated_rotation,estimated_translation

    def __getitem__(self, index):
        data_dict = {}

        # metadata
        metadata: Dict = self.metadata_list[index]
        data_dict['scene_name'] = metadata['scene_name']
        data_dict['ref_frame'] = metadata['frag_id0']
        data_dict['src_frame'] = metadata['frag_id1']
        data_dict['overlap'] = metadata['overlap']

        # get transformation
        rotation = metadata['rotation']
        translation = metadata['translation']

        # get point cloud
        ref_points = self._load_point_cloud(metadata['pcd0'])
        src_points = self._load_point_cloud(metadata['pcd1'])

        # get prior data
        ref_id = data_dict['ref_frame']
        src_id = data_dict['src_frame']
        scene_name = data_dict['scene_name']
        if self.using_geo_prior:
            if self.use_augmentation:
                file_name = osp.join(self.geo_train_prior, scene_name, f'{ref_id}_{src_id}.npz')
            else:
                file_name = osp.join(self.geo_prior, scene_name, f'{ref_id}_{src_id}.npz')
            prior_data = np.load(file_name)
            estimated_transform=prior_data['estimated_transform']
            estimated_rotation=estimated_transform[:3,:3]
            estimated_translation=estimated_transform[:3,3]
            # src_aligned_pcd=np.matmul(rotation,src_points.T).T+translation.T
            # draw_registration_result(src_aligned_pcd,ref_points)

        # augmentation
        if self.use_augmentation:
            aug_src=np.random.rand(1)[0]
            ref_points, src_points, rotation, translation,rot_ab,estimated_rotation,estimated_translation = self._augment_point_cloud(
                ref_points, src_points, rotation, translation,aug_src,estimated_rotation,estimated_translation
            )
        # src_aligned_pcd=np.matmul(rotation,src_points.T).T+translation.T
        # draw_registration_result(src_aligned_pcd,ref_points)
        if self.rotated:
            ref_rotation = random_sample_rotation_v2()
            ref_points = np.matmul(ref_points, ref_rotation.T)
            rotation = np.matmul(ref_rotation, rotation)
            translation = np.matmul(ref_rotation, translation)

            src_rotation = random_sample_rotation_v2()
            src_points = np.matmul(src_points, src_rotation.T)
            rotation = np.matmul(rotation, src_rotation.T)

        transform = get_transform_from_rotation_translation(rotation, translation)

        # get geo_prior
        if self.using_geo_prior:
            estimated_transform = get_transform_from_rotation_translation(estimated_rotation, estimated_translation)
            # file_name = osp.join(self.geo_prior, scene_name, f'{ref_id}_{src_id}.npz')
            # prior_data = np.load(file_name)
            corr_indices = get_correspondences(ref_points, src_points, estimated_transform, 0.0375)
            if len(corr_indices)<self.overlap_min_points and self.use_augmentation:
                estimated_transform=transform
            if 'est_pose_src_idx_f' in prior_data.keys():
                est_pose_ref_idx_f = prior_data['est_pose_ref_idx_f'][:-1]
                est_pose_src_idx_f = prior_data['est_pose_src_idx_f'][:-1]
            # if est_pose_ref_idx_f.shape[0]<100 or est_pose_src_idx_f.shape[0]<100:
                data_dict['est_ref_idx_f'] = est_pose_ref_idx_f
                data_dict['est_src_idx_f'] = est_pose_src_idx_f
            elif 'ref_corr_indices' in prior_data.keys():
                # ref_corr_indices_f = prior_data['ref_corr_indices_f']
                # src_corr_indices_f = prior_data['src_corr_indices_f']
                # data_dict['est_ref_idx_f'] = ref_corr_indices_f
                # data_dict['est_src_idx_f'] = src_corr_indices_f
                ref_corr_indices = prior_data['ref_corr_indices']
                src_corr_indices = prior_data['src_corr_indices']
                data_dict['ref_corr_indices'] = ref_corr_indices
                data_dict['src_corr_indices'] = src_corr_indices
            # draw_registration_result(src_points[src_corr_indices,:],ref_points[ref_corr_indices,:],estimated_transform)
            data_dict['estimated_transform'] = estimated_transform
            del prior_data

        # get correspondences
        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)

        return data_dict
