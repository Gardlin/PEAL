import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from geotransformer.utils.common import ensure_dir


_C = edict()

# common
_C.seed = 7351

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
_C.output_dir = osp.join(_C.root_dir, 'output', _C.exp_name)
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
_C.log_dir = osp.join(_C.output_dir, 'logs')
_C.event_dir = osp.join(_C.output_dir, 'events')
_C.feature_dir = osp.join(_C.output_dir, 'features')
_C.registration_dir = osp.join(_C.output_dir, 'registration')

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)
ensure_dir(_C.feature_dir)
ensure_dir(_C.registration_dir)

# data
_C.data = edict()
_C.data.dataset_root = osp.join(_C.root_dir, 'data', '3DMatch')
_C.data.dataset_root = '/home/levi/yjl/pcrcg_data/geotransformer_color/Geotransformer_implicit_color/data/3DMatch'

# train data
_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 8
_C.train.point_limit = 30000
_C.train.use_augmentation = True
_C.train.augmentation_noise = 0.005
_C.train.augmentation_rotation = 1.0

# test data
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 8
_C.test.point_limit = None


# prior data
_C.train.image_path= '/home/lab507/data/3img_3dmatch/3dmatch_image'
_C.train.geo_train_prior= '/home/levi/yjl/PEAL-V1/data_prior/geo_prior_train_epoch-15-rre19.287'
_C.train.geo_prior= '/home/levi/yjl/PEAL-V1/data_prior/geo_prior_testval'
_C.test.using_iter_prior = True
_C.train.prior_min_points = 100
_C.train.overlap_min_points = 50
_C.train.superglue_prior=  '/home/lab507/data/SuperGluePretrainedNetwork-master/dump_match_pairs_160_2img'
_C.train.using_geo_prior=True
_C.train.using_2d_prior=True
_C.train.superglue_window_size=5




# evaluation
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 0.1
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rmse_threshold = 0.2
_C.eval.rre_threshold = 15.0
_C.eval.rte_threshold = 0.3

# ransac
_C.ransac = edict()
_C.ransac.distance_threshold = 0.05
_C.ransac.num_points = 3
_C.ransac.num_iterations = 1000

# optim
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 1
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 42
_C.optim.grad_acc_steps = 1

# model - backbone
_C.backbone = edict()
_C.backbone.num_stages = 4
_C.backbone.init_voxel_size = 0.025
_C.backbone.kernel_size = 15
_C.backbone.base_radius = 2.5
_C.backbone.base_sigma = 2.0
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.backbone.group_norm = 32
_C.backbone.input_dim = 1
_C.backbone.init_dim = 64
_C.backbone.output_dim = 256

# model - Global
_C.model = edict()
_C.model.ground_truth_matching_radius = 0.05
_C.model.num_points_in_patch = 64
_C.model.num_sinkhorn_iterations = 100

# model - Coarse Matching
_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128
_C.coarse_matching.overlap_threshold = 0.1
_C.coarse_matching.num_correspondences = 256
_C.coarse_matching.dual_normalization = True

# model - GeoTransformer
_C.geotransformer = edict()
_C.geotransformer.input_dim = 1024
_C.geotransformer.hidden_dim = 256
_C.geotransformer.output_dim = 256
_C.geotransformer.num_heads = 4
_C.geotransformer.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
_C.geotransformer.sigma_d = 0.2
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = 'max'

# model - Fine Matching
_C.fine_matching = edict()
_C.fine_matching.topk = 3
_C.fine_matching.acceptance_radius = 0.1
_C.fine_matching.mutual = True
_C.fine_matching.confidence_threshold = 0.05
_C.fine_matching.use_dustbin = False
_C.fine_matching.use_global_score = False
_C.fine_matching.correspondence_threshold = 3
_C.fine_matching.correspondence_limit = None
_C.fine_matching.num_refinement_steps = 5

# loss - Coarse level
_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 24
_C.coarse_loss.positive_overlap = 0.1

# loss - Fine level
_C.fine_loss = edict()
_C.fine_loss.positive_radius = 0.05

# loss - Overall
_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_loss = 1.0


def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args


def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, 'output')


if __name__ == '__main__':
    main()
