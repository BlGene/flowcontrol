import os
import copy
import numpy as np

import open3d as o3d
from recording_loader import RecordingLoader
from pdb import set_trace


class RegistrationModule:
    def __init__(self, desc="Evaluation Epoch", size=None):
        """
        self._args = args
        self._desc = desc

        # Set random seed, possibly on Cuda
        config.configure_random_seed(args)

        # Configure model and loss
        model_and_loss = config.configure_model_and_loss(args)
        """
        pass

    @staticmethod
    def preprocess_point_cloud(pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    @staticmethod
    def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                         target_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.5
        print(":: Apply fast global registration with distance threshold %.3f" \
                % distance_threshold)
        result = o3d.registration.registration_fast_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        return result

    # this is a baseline method
    @staticmethod
    def execute_global_registration(source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(False), 4, [
                o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
        return result

    @staticmethod
    def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])


    """
    Comput the registration of pcd1 to pcd2
    Input:
        pcd1 the segmented demonstration pointcloud
        pcd2 the full live observed pointcould
    Output:
        transformation the relative trasnformation as matrix
    """
    def register(self, pcd1_arr, pcd2_arr):
        # Next get two point clouds, then register
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pcd1_arr[:, :3])
        pcd1.colors = o3d.utility.Vector3dVector(pcd1_arr[:, 4:7]/255.)
        #o3d.visualization.draw_geometries([pcd1])

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pcd2_arr[:, :3])
        pcd2.colors = o3d.utility.Vector3dVector(pcd2_arr[:, 4:7]/255.)
        # plot pointclouds
        #o3d.visualization.draw_geometries([pcd1, pcd2])

        import time
        source = pcd1
        target = pcd2
        voxel_size = 0.005  # means 5mm for the dataset
        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)

        #start = time.time()
        #result_ransac = self.execute_global_registration(source_down, target_down,
        #                                                 source_fpfh, target_fpfh,
        #                                                 voxel_size)
        #print("Global registration took %.3f sec.\n" % (time.time() - start))
        #print(result_ransac)
        #self.draw_registration_result(source_down, target_down,
        #                         result_ransac.transformation)

        #start = time.time()
        result_fast = self.execute_fast_global_registration(source_down, target_down,
                                                           source_fpfh, target_fpfh,
                                                           voxel_size)
        #print("Fast global registration took %.3f sec.\n" % (time.time() - start))
        #print(result_fast)
        #self.draw_registration_result(source_down, target_down,
        #                              result_fast.transformation)

        return result_fast

def test_corr_module():
    test_dir = "/home/argusm/lang/gym_grasping/gym_grasping/flow_control/pose_estimation_data/"
    object = "wd_40"
    rec = RecordingLoader(test_dir, object)
    # do this after loading data to allow parameterization
    reg = RegistrationModule(desc="Deploy")

    pcd1_arr = rec.get_pointcloud(0, masked=True)
    pcd2_arr = rec.get_pointcloud(1)
    reg_res = reg.register(pcd1_arr, pcd2_arr)
    print("done.")

if __name__ == "__main__":
    test_corr_module()
