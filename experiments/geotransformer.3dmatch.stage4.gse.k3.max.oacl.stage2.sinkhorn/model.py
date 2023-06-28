import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import numpy as np
from geotransformer.modules.ops import pairwise_distance
from geotransformer.datasets.registration.threedmatch.dataset import get_correspondences
import numpy as np
from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)

from backbone import KPConvFPN


class GeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius
        self.using_geo_prior=cfg.train.using_geo_prior
        self.prior_min_points=cfg.train.prior_min_points
        self.using_2d_prior=cfg.train.using_2d_prior
        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

    def forward(self, data_dict):
        output_dict = {}

        # Downsample point clouds
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        # 1. Generate ground truth node correspondences
        ref_point_to_node, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        src_point_to_node, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        if self.using_geo_prior:
            estimated_transform = data_dict['estimated_transform'].detach()
            # corr_indices = get_correspondences(ref_points.detach().cpu().numpy(), src_points.detach().cpu().numpy(), estimated_transform.detach().cpu().numpy(), 0.0375)
            # est_src_idx =np.array (list(set(corr_indices[:,1].tolist())))
            # est_tgt_idx =np.array( list(set(corr_indices[:,0].tolist())))
            corr_indices_f = get_correspondences(ref_points_f.detach().cpu().numpy(), src_points_f.detach().cpu().numpy(), estimated_transform.detach().cpu().numpy(), 0.0375)
            est_ref_idx_f =np.array( list(set(corr_indices_f[:,0].tolist())))
            est_src_idx_f =np.array (list(set(corr_indices_f[:,1].tolist())))
            # from geotransformer.utils.registration import compute_overlap
            # origin_overlap=compute_overlap(src_points=src_points.detach().cpu().numpy(),ref_points=ref_points.detach().cpu().numpy(),transform=transform.detach().cpu().numpy())
            # geo_est_overlap=compute_overlap(src_points=src_points[est_src_idx,:].detach().cpu().numpy(),ref_points=ref_points[est_tgt_idx,:].detach().cpu().numpy(),transform=estimated_transform_baseline.detach().cpu().numpy())

            if est_src_idx_f.shape[0]<self.prior_min_points or est_ref_idx_f.shape[0]<self.prior_min_points:
                if self.training:
                    corr_indices_f = get_correspondences(ref_points_f.detach().cpu().numpy(), src_points_f.detach().cpu().numpy(), transform.detach().cpu().numpy(), 0.0375)
                    est_ref_idx_f =np.array( list(set(corr_indices_f[:len(corr_indices_f)//2,0].tolist())))
                    est_src_idx_f =np.array (list(set(corr_indices_f[:len(corr_indices_f)//2,1].tolist())))
                elif 'ref_corr_indices' in data_dict.keys():
                    ref_corr_indices =data_dict['ref_corr_indices'][:self.prior_min_points*10]
                    src_corr_indices =data_dict['src_corr_indices'][:self.prior_min_points*10]
                    ref_prior_corr_points=ref_points[ref_corr_indices,:]
                    src_prior_corr_points=src_points[src_corr_indices,:]
                    k = 0
                    src_dist_map_f = torch.sqrt(pairwise_distance(src_prior_corr_points.unsqueeze(0), src_points_f.unsqueeze(0)))  # (B, N, N)
                    est_src_idx_f = src_dist_map_f.topk(k=k + 1, dim=2, largest=False)[1][0, :, 0]  # (B, N, k)
                    ref_dist_map_f = torch.sqrt(pairwise_distance(ref_prior_corr_points.unsqueeze(0), ref_points_f.unsqueeze(0)))  # (B, N, N)
                    est_ref_idx_f = ref_dist_map_f.topk(k=k + 1, dim=2, largest=False)[1][0, :, 0]  # (B, N, k)
                else:
                    est_ref_idx_f =np.random.permutation(ref_points_f.shape[0])[: self.prior_min_points]
                    est_src_idx_f =np.random.permutation(src_points_f.shape[0])[: self.prior_min_points]

            # est_ref_idx_f,est_src_idx_f=torch.unique(est_ref_idx_f),torch.unique(est_src_idx_f)
            # indices = np.random.permutation(src_points_c.shape[0])[: 50]
            # points = points[indices]
            ref_overlapped_points_c_idx=ref_point_to_node[est_ref_idx_f]
            src_overlapped_points_c_idx=src_point_to_node[est_src_idx_f]
            src_overlapped_points_c_idx,ref_overlapped_points_c_idx=torch.unique(src_overlapped_points_c_idx),torch.unique(ref_overlapped_points_c_idx)
            src_points_c_idx=torch.arange(src_points_c.shape[0]).to(src_overlapped_points_c_idx.device)
            ref_points_c_idx=torch.arange(ref_points_c.shape[0]).to(src_overlapped_points_c_idx.device)
            src_no_overlapped_points_c_list=[i.item()  for i in src_points_c_idx if i not in src_overlapped_points_c_idx  ]
            src_no_overlapped_points_c_idx=torch.from_numpy(np.array(src_no_overlapped_points_c_list))
            ref_no_overlapped_points_c_list=[i.item()  for i in ref_points_c_idx if i not in ref_overlapped_points_c_idx  ]
            ref_no_overlapped_points_c_idx=torch.from_numpy(np.array(ref_no_overlapped_points_c_list))
        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_c, src_feats_c, ref_embeddings , src_embeddings = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
            ref_overlapped_points_c_idx,src_overlapped_points_c_idx,
            ref_no_overlapped_points_c_idx,src_no_overlapped_points_c_idx
        )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform

        return output_dict


def create_model(config):
    model = GeoTransformer(config)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
