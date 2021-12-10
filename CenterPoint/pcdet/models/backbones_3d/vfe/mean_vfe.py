import torch

from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']

        voxels_prm,voxel_num_points_prm = batch_dict['voxels_prm'], batch_dict['voxel_num_points_prm']
        if len(voxels_prm.shape)==4:
            voxels_prm = voxels_prm[0]
        #print(f"voxel features in mean_vfe  {voxel_features.shape}")
        #prm = voxel_features
        #voxel_features = voxel_features[:,:, :-1]

        #print(f"voxel features in mean_vfe 2 {voxel_features.shape}")
        #print(f"voxel prm in mean_vfe  {prm.shape}")

        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()


        prm_mean = voxels_prm[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points_prm.view(-1, 1), min=1.0).type_as(voxels_prm)
        prm_mean = prm_mean / normalizer
        batch_dict['voxels_prm'] = prm_mean.contiguous()
        
        assert points_mean.shape[0] == prm_mean.shape[0], (points_mean.shape[0], prm_mean.shape[0])
        #batch_dict['voxel_prm'] = prm_mean.contiguous()
        #print(f"voxel prm in mean_vfe  {prm.shape}")
        return batch_dict
