import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation

# in practice, it is better to scan point cloud with more curvature region than plannar region, and the number of point cloud fragments should be larger than 3

def pose_to_transformation_matrix(pose):
    x, y, z, ox, oy, oz, ow = pose
    rotation_matrix = Rotation.from_quat([ox, oy, oz, ow]).as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [x, y, z]
    return transformation_matrix

def transform_point_cloud(pcd, transformation_matrix):
    return pcd.transform(transformation_matrix)

def denoise_point_cloud(pcd, nb_neighbors=30, std_ratio=1.5):
    pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd_clean

def preprocess_point_cloud(pcd, voxel_size):
    print("Downsampling and denoise scanned pcl, then ppplying FPFH algo to get local geometric feature descriptors")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down = denoise_point_cloud(pcd_down)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    
    return pcd_down, pcd_fpfh

def merge_point_clouds(ply_files, camera_poses):
    merged = o3d.geometry.PointCloud()
    for ply_file, pose in zip(ply_files, camera_poses):
        pcd = o3d.io.read_point_cloud(ply_file)
        transformation_matrix = pose_to_transformation_matrix(pose)
        pcd = transform_point_cloud(pcd, transformation_matrix)
        merged += pcd
    return merged

def print_pose_in_rpy(transformation):
    """
    Print translation and orientation in roll-pitch-yaw.
    """
    translation = transformation[:3, 3]
    # Make a writable copy of the rotation matrix
    rotation_matrix = transformation[:3, :3].copy()
    rotation = Rotation.from_matrix(rotation_matrix)
    rpy = rotation.as_euler('xyz', degrees=False)
    print("Translation (x, y, z):", translation)
    print("Rotation (r, p, y) in degrees:", rpy)
    ground_truth_Translation = [1.38494077,-0.07737339,0.95311693]
    ground_truth_Rotation = [0.00784205,0.016702,1.64769838]
    print("Ground truth Translation is:", ground_truth_Translation)
    print("Ground truth Rotation is:", ground_truth_Rotation)
    translation_error = np.abs(translation - ground_truth_Translation).mean()
    rotation_error = np.abs(rpy - ground_truth_Rotation).mean()

    print(f"Translation error (average): {translation_error:.4f}")
    print(f"Rotation error (average): {rotation_error:.4f}")
    
def estimate_object_pose_from_scans(ply_files, camera_poses, mesh_path, voxel_size=0.02, loop_time=10, visualize=True):
    print("There are %d point cloud fragments for registration"%len(ply_files))
    merged_pcd = merge_point_clouds(ply_files, camera_poses)
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh_sampled = mesh.sample_points_poisson_disk(5000)

    pcd_down, pcd_fpfh = preprocess_point_cloud(merged_pcd, voxel_size)
    mesh_down, mesh_fpfh = preprocess_point_cloud(mesh_sampled, voxel_size)


    best_result_icp = None
    best_fitness = -1

    # loop N times for ransac + icp to get best fit result
    for i in range(loop_time):
        print("Conducting Ransac + icp pcl registration algorithm, loop number ", i+1)

        # Ransac (globally coarse registration)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=mesh_down, target=pcd_down,
            source_feature=mesh_fpfh, target_feature=pcd_fpfh,
            mutual_filter=True,
            max_correspondence_distance=voxel_size * 1.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000)
        )
        # icp (local refinement after Ransac)
        result_icp = o3d.pipelines.registration.registration_icp(
            mesh_down, pcd_down, voxel_size * 0.4,
            result.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        
        if result_icp.fitness > best_fitness:
            best_fitness = result_icp.fitness
            best_result_icp = result_icp

    if visualize:
        # print("\nEstimated transformation matrix:\n", best_result_icp.transformation)
        print_pose_in_rpy(best_result_icp.transformation)
        mesh_transformed = mesh.transform(best_result_icp.transformation)
        o3d.visualization.draw_geometries([merged_pcd, mesh_transformed])

    return best_result_icp.transformation


# full sets of 12 pcl fragments (bad effeciency and stable good result)
scanned_pcd_files =["data/2025_03_14/ply_00-53-55.ply","data/2025_03_14/ply_00-54-01.ply","data/2025_03_14/ply_00-54-07.ply","data/2025_03_14/ply_00-54-15.ply"
                    ,"data/2025_03_14/ply_00-54-22.ply","data/2025_03_14/ply_00-54-29.ply","data/2025_03_14/ply_00-54-35.ply","data/2025_03_14/ply_00-54-42.ply"
                    ,"data/2025_03_14/ply_00-54-48.ply","data/2025_03_14/ply_00-54-54.ply","data/2025_03_14/ply_00-55-03.ply","data/2025_03_14/ply_00-55-12.ply"]
camera_poses = [ [0.505679 ,0.372129 ,0.874785 ,0.679306 ,  0.673188, 0.002037 , 0.292160]
                ,[0.485921 ,0.545722 ,0.797907 ,0.677711 ,  0.615950, -0.054073, 0.397982]
                ,[0.726444 ,0.519243 ,0.900194 ,0.649884 ,  0.704604, -0.198384, 0.204521]
                ,[0.423012 ,0.130172 ,0.959621 ,0.673691 ,  0.694444, 0.089513 , 0.236380]
                ,[0.692647 ,-0.103832, 1.038555, 0.679308,  0.733730, -0.001802, 0.013340]
                ,[0.941126 ,-0.233288, 1.001513, 0.678050,  0.710926, -0.084144, -0.166594]
                ,[0.593556 ,-0.566070, 0.974923, 0.630377,  0.729540, 0.251736 ,-0.083812]
                ,[0.524621 ,-0.613491, 0.951180, 0.614651,  0.732016, 0.288683 ,-0.054935]
                ,[0.213177 ,-0.674643, 0.644158, 0.390709,  0.695334, 0.571802 ,0.192090]
                ,[0.109050 ,-0.521054, 0.602719, 0.500031,  0.631879, 0.480066 ,0.346748]
                ,[1.011881 ,-0.817526, 0.630772, 0.691943,  0.473837, 0.028720 ,-0.543938]
                ,[0.735649 ,-0.941664, 0.046652, -0.412074, 0.289148, 0.607811 ,0.614129]]


# # partial sets of 5 pcl fragments (good effeciency and best& stable registration result)
# scanned_pcd_files = ["ply_00-53-55.ply", "ply_00-54-22.ply","ply_00-54-42.ply","ply_00-54-48.ply","ply_00-55-12.ply"]

# camera_poses = [
#     [0.505679, 0.372129, 0.874785, 0.679306, 0.673188, 0.002037, 0.292160],
#     [0.692647 ,-0.103832, 1.038555, 0.679308,  0.733730, -0.001802, 0.013340],
#     [0.524621 ,-0.613491, 0.951180, 0.614651,  0.732016, 0.288683 ,-0.054935],
#     [0.213177 ,-0.674643, 0.644158, 0.390709,  0.695334, 0.571802 ,0.192090],
#     [0.735649 ,-0.941664, 0.046652, -0.412074, 0.289148, 0.607811 ,0.614129]
# ]


# # partial sets of 3 pcl fragments(best effeciency but bad registration result)
# scanned_pcd_files = [ "ply_00-54-22.ply","ply_00-54-42.ply","ply_00-54-48.ply"]

# camera_poses = [
#     [0.692647 ,-0.103832, 1.038555, 0.679308,  0.733730, -0.001802, 0.013340],
#     [0.524621 ,-0.613491, 0.951180, 0.614651,  0.732016, 0.288683 ,-0.054935],
#     [0.213177 ,-0.674643, 0.644158, 0.390709,  0.695334, 0.571802 ,0.192090],
# ]


# # partial sets of 1 pcl fragments (worst registration result)
# scanned_pcd_files = ["ply_00-54-48.ply"]

# camera_poses = [
#     [0.213177 ,-0.674643, 0.644158, 0.390709,  0.695334, 0.571802 ,0.192090],
# ]

object_mesh_file = "data/8400310XKM42A_new.STL"

T_base_object = estimate_object_pose_from_scans(scanned_pcd_files, camera_poses, object_mesh_file)