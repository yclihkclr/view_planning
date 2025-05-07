import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d

viewpoint_6Ds_list = []

viewpoint_6Ds = [(np.array([-0.03439183,  0.70912371,  0.02172246]), np.array([[ 0.99861581,  0.        ,  0.05259716],
       [ 0.00241911, -0.99894175, -0.0459296 ],
       [ 0.0525415 ,  0.04599326, -0.99755903]])), (np.array([ 0.44560817,  0.80912371, -0.07827754]), np.array([[ 0.94416368,  0.        , -0.32947676],
       [-0.11487248, -0.93725254, -0.32918382],
       [-0.30880293,  0.34865122, -0.88491981]])), (np.array([-0.51439183,  0.48912371, -0.13827754]), np.array([[ 0.79984612,  0.        ,  0.60020512],
       [-0.17331395, -0.95740216,  0.23096186],
       [ 0.57463768, -0.28875786, -0.7657744 ]])), (np.array([-0.19439183,  1.22912371, -0.53827754]), np.array([[ 0.98971321,  0.        , -0.14306558],
       [-0.14289199, -0.04924652, -0.98851235],
       [-0.00704548,  0.99878665, -0.04873993]])), (np.array([-0.71439183,  0.44912371, -0.45827754]), np.array([[ 0.44913151,  0.        ,  0.89346566],
       [-0.60660424, -0.73419928,  0.30493067],
       [ 0.65598184, -0.67893403, -0.32975203]])), (np.array([ 0.62560817,  0.80912371, -0.13827754]), np.array([[ 0.87191406,  0.        , -0.48965893],
       [-0.17258234, -0.93582907, -0.30730976],
       [-0.45823706,  0.35245419, -0.81596252]])), (np.array([-0.13439183,  0.56912371, -0.95827754]), np.array([[ 0.99125138,  0.        , -0.13198754],
       [ 0.03290894,  0.96841764,  0.24715233],
       [ 0.12781906, -0.24933366,  0.95994532]])), (np.array([-0.77439183,  0.66912371, -0.81827754]), np.array([[ 0.24381952,  0.        ,  0.96982062],
       [-0.25962426,  0.96350137,  0.06527131],
       [-0.9344235 , -0.26770339,  0.23492045]])), (np.array([-0.57439183,  1.04912371, -0.29827754]), np.array([[ 0.79436209,  0.        ,  0.60744454],
       [ 0.47357621, -0.62625227, -0.61930095],
       [ 0.38041353,  0.77962048, -0.49747106]])), (np.array([ 0.08560817,  0.32912371, -0.79827754]), np.array([[ 0.99993805,  0.        , -0.01113061],
       [ 0.00539718,  0.87457206,  0.48486552],
       [ 0.00973452, -0.48489556,  0.87451789]])), (np.array([ 0.38560817,  1.00912371, -0.11827754]), np.array([[ 0.97389658,  0.        , -0.22699218],
       [-0.12836811, -0.8247361 , -0.55075585],
       [-0.18720864,  0.56551779, -0.80320766]])), (np.array([ 0.00560817,  1.06912371, -0.77827754]), np.array([[ 0.97668352, -0.        , -0.21468421],
       [-0.12359476,  0.81765754, -0.5622815 ],
       [ 0.17553816,  0.57570492,  0.79859264]])), (np.array([-0.51439183,  0.86912371, -0.07827754]), np.array([[ 0.90977134,  0.        ,  0.41510976],
       [ 0.11619289, -0.96002659, -0.25465303],
       [ 0.3985164 ,  0.27990883, -0.87340467]])), (np.array([-0.17439183,  0.48912371,  0.00172246]), np.array([[ 0.97454658,  0.        ,  0.22418511],
       [-0.06019859, -0.96327361,  0.26168701],
       [ 0.2159516 , -0.26852181, -0.938755  ]])), (np.array([0.32560817, 0.72912371, 0.00172246]), np.array([[ 0.97499243,  0.        , -0.22223809],
       [-0.04960052, -0.97477578, -0.21760507],
       [-0.21663231,  0.22318642, -0.950399  ]])), (np.array([-0.61439183,  0.40912371, -0.65827754]), np.array([[ 0.86881947,  0.        ,  0.495129  ],
       [-0.49512136, -0.00555366,  0.86880607],
       [ 0.00274978, -0.99998458, -0.00482512]])), (np.array([-0.19439183,  1.10912371, -0.23827754]), np.array([[ 0.99364418,  0.        , -0.11256665],
       [-0.08076923, -0.69653417, -0.71296317],
       [-0.07840652,  0.71752362, -0.69210713]])), (np.array([-0.27439183,  0.22912371, -0.45827754]), np.array([[ 9.87699712e-01,  0.00000000e+00,  1.56362651e-01],
       [-1.56362030e-01,  2.81792719e-03,  9.87695791e-01],
       [-4.40618566e-04, -9.99996030e-01,  2.78326588e-03]])), (np.array([ 0.70560817,  0.92912371, -0.25827754]), np.array([[ 0.79534934,  0.        , -0.60615132],
       [-0.34423255, -0.82309846, -0.45167785],
       [-0.49892222,  0.5678987 , -0.65465082]])), (np.array([-0.23439183,  0.96912371, -0.83827754]), np.array([[ 0.98486336, -0.        ,  0.17333253],
       [ 0.06400521,  0.92932511, -0.36367318],
       [-0.16108227,  0.36926257,  0.91525824]])), (np.array([-0.27439183,  0.74912371, -0.01827754]), np.array([[ 0.91898058,  0.        ,  0.3943028 ],
       [ 0.01738299, -0.99902777, -0.04051361],
       [ 0.39391944,  0.04408538, -0.91808712]])), (np.array([ 0.02560817,  0.94912371, -0.85827754]), np.array([[ 0.97912536, -0.        , -0.2032573 ],
       [-0.05677816,  0.96019187, -0.27351016],
       [ 0.195166  ,  0.27934131,  0.94014821]])), (np.array([-0.65439183,  0.42912371, -0.35827754]), np.array([[ 0.52433846,  0.        ,  0.85150994],
       [-0.53260646, -0.78023642,  0.32796569],
       [ 0.66437907, -0.62548472, -0.40910796]])), (np.array([-0.67439183,  0.62912371, -0.99827754]), np.array([[ 0.60948218,  0.        ,  0.79279977],
       [-0.21916675,  0.9610293 ,  0.16848924],
       [-0.7619038 , -0.27644654,  0.58573024]])), (np.array([-0.03439183,  0.96912371, -0.07827754]), np.array([[ 0.99479387,  0.        , -0.10190762],
       [-0.05389037, -0.84873656, -0.52606281],
       [-0.08649272,  0.5288159 , -0.84431792]])), (np.array([-0.41439183,  1.08912371, -0.23827754]), np.array([[ 0.97011664,  0.        ,  0.24263903],
       [ 0.174009  , -0.69691708, -0.69572082],
       [ 0.16909929,  0.71715172, -0.67609086]])), (np.array([ 0.20560817,  0.98912371, -0.09827754]), np.array([[ 0.99997837,  0.        ,  0.00657752],
       [ 0.00328949, -0.86596182, -0.50009929],
       [ 0.00569589,  0.50011011, -0.86594309]])), (np.array([-0.09439183,  0.42912371, -0.07827754]), np.array([[ 0.99856478,  0.        ,  0.05355735],
       [-0.02545456, -0.87983641,  0.47459452],
       [ 0.04712171, -0.47527665, -0.87857365]])), (np.array([ 0.72560817,  0.60912371, -0.13827754]), np.array([[ 0.79585045,  0.        , -0.60549324],
       [-0.05566836, -0.99576465, -0.07316959],
       [-0.60292877,  0.09193887, -0.79247974]])), (np.array([-0.25439183,  0.80912371, -0.05827754]), np.array([[ 0.96543008,  0.        ,  0.26066215],
       [ 0.09011628, -0.93833749, -0.33376911],
       [ 0.24458906,  0.34572064, -0.90589924]])), (np.array([ 0.00560817,  0.88912371, -0.03827754]), np.array([[ 0.9999342 ,  0.        , -0.01147177],
       [-0.00436542, -0.92476621, -0.38051072],
       [-0.01060871,  0.38053576, -0.92470536]])), (np.array([-0.41439183,  0.66912371, -0.11827754]), np.array([[ 0.91188133,  0.        ,  0.41045395],
       [ 0.03204392, -0.99694792, -0.07119008],
       [ 0.40920121,  0.07806945, -0.9090982 ]])), (np.array([-0.23439183,  0.66912371, -0.05827754]), np.array([[ 0.92721031,  0.        ,  0.3745411 ],
       [-0.01687107, -0.99898497,  0.04176586],
       [ 0.37416093, -0.04504464, -0.92626917]])), (np.array([ 0.18560817,  0.94912371, -0.03827754]), np.array([[ 0.99830767,  0.        , -0.05815315],
       [-0.02480831, -0.90443896, -0.4258811 ],
       [-0.05259598,  0.42660305, -0.90290836]])), (np.array([-0.17439183,  0.54912371, -0.93827754]), np.array([[ 0.9941684 ,  0.        , -0.10783873],
       [ 0.03937449,  0.93095898,  0.36299454],
       [ 0.10039344, -0.36512379,  0.92553   ]])), (np.array([-0.25439183,  0.58912371, -0.05827754]), np.array([[ 0.9147277 ,  0.        ,  0.40407083],
       [-0.08261035, -0.97887801,  0.1870117 ],
       [ 0.39553605, -0.20444521, -0.89540683]])), (np.array([ 0.06560817,  0.36912371, -0.85827754]), np.array([[ 0.9713231 ,  0.        , -0.2377634 ],
       [ 0.090402  ,  0.92489678,  0.36931481],
       [ 0.21990661, -0.38021829,  0.89837361]]))]

if len(viewpoint_6Ds_list) == 0:
    print("viewpoint_6Ds_list is not given, driven from viewpoint_6Ds")
    for position, rotation_matrix in viewpoint_6Ds:
        # Convert rotation matrix to quaternion
        quaternion = Rotation.from_matrix(rotation_matrix).as_quat()  # Returns (ox, oy, oz, ow)
        
        # Store in the required format
        x, y, z = position
        ox, oy, oz, ow = quaternion  # Quaternion as (x, y, z, w)
        viewpoint_6Ds_list.append([x, y, z, ox, oy, oz, ow])
else:
    print("viewpoint_6Ds_list is given")

# Print the result, it is oTc
print(f"viewpoint_6Ds_list are: {viewpoint_6Ds_list}")



# Given transformation from base to object
base_to_object_translation = np.array([1.38494077,-0.07737339,0.95311693])  # (x, y, z)
base_to_object_rpy = np.array([0.00784205,0.016702,1.64769838])  # (roll, pitch, yaw)

# Convert RPY to rotation matrix
base_to_object_rotation = Rotation.from_euler('xyz', base_to_object_rpy).as_matrix()

# Construct homogeneous transformation matrix (4x4)
T_base_object = np.eye(4)
T_base_object[:3, :3] = base_to_object_rotation
T_base_object[:3, 3] = base_to_object_translation

# Transform viewpoints from object frame to base frame
viewpoint_6Ds_list_base = []

for pose in viewpoint_6Ds_list:
    # Convert position to homogeneous coordinates
    position_h = np.array([pose[0], pose[1], pose[2], 1])  # (x, y, z, 1)

    # Transform position
    transformed_position_h = T_base_object @ position_h
    transformed_position = transformed_position_h[:3]  # Extract (x, y, z)

    # Transform rotation using quaternion multiplication
    object_quaternion = Rotation.from_quat([pose[3],pose[4],pose[5],pose[6]])  # (ox, oy, oz, ow)
    base_rotation = Rotation.from_matrix(base_to_object_rotation)  # Object to base rotation
    transformed_quaternion = base_rotation * object_quaternion  # Quaternion multiplication

    # Convert to list format
    transformed_pose = list(transformed_position) + list(transformed_quaternion.as_quat())
    viewpoint_6Ds_list_base.append(transformed_pose)

# Print transformed viewpoints
print(f"viewpoint_6Ds_list_base are: {viewpoint_6Ds_list_base}")


# Base Z-axis (upward direction)
base_z_axis = np.array([0, 0, 1])

# Filtered list of viewpoints
filtered_viewpoint_6Ds_list_base = []

for pose in viewpoint_6Ds_list_base:
    x, y, z, ox, oy, oz, ow = pose

    # Convert quaternion to rotation matrix to extract Z-axis
    rotation = Rotation.from_quat([ox, oy, oz, ow])
    viewpoint_z_axis = rotation.as_matrix()[:, 2]  # Extract the third column (Z-axis)

    # Compute dot product with the base Z-axis
    dot_product = np.dot(viewpoint_z_axis, base_z_axis)

    # Keep the viewpoint if the dot product is negative (i.e., looking downward) 45degree
    if dot_product < 0.5:
        filtered_viewpoint_6Ds_list_base.append(pose)

print(f"Original viewpoints number: {len(viewpoint_6Ds_list_base)}, Filtered viewpoints number: {len(filtered_viewpoint_6Ds_list_base)}")


def adjust_viewpoints_if_too_close(object_mesh_path, filtered_viewpoint_6Ds_list_base, min_distance=0.45):
    """
    Adjust viewpoints that are too close to the object by shifting them along their negative Z-axis.
    
    Parameters:
        object_mesh_path (str): Path to the object's STL file.
        filtered_viewpoint_6Ds_list_base (list): List of viewpoint poses in base frame as [x, y, z, ox, oy, oz, ow].
        min_distance (float): Minimum allowable distance between viewpoint and object.
        offset_distance (float): Distance to move the viewpoint backward if it's too close.
    
    Returns:
        list: Adjusted viewpoint poses.
    """

    # Load object mesh
    object_mesh = o3d.io.read_triangle_mesh(object_mesh_path)

    object_mesh.transform(T_base_object)
    
    # Convert mesh to point cloud (sample points on the surface)
    object_pcd = object_mesh.sample_points_uniformly(number_of_points=100000)
    
    # Build KDTree for fast nearest neighbor search
    object_kdtree = o3d.geometry.KDTreeFlann(object_pcd)

    nearest_distances = []
    adjusted_viewpoints = []
    
    for vp in filtered_viewpoint_6Ds_list_base:
        x, y, z, ox, oy, oz, ow = vp

        # Find the nearest point on the object surface
        viewpoint_position = np.array([x, y, z])
        [_, idx, _] = object_kdtree.search_knn_vector_3d(viewpoint_position, 1)
        nearest_point = np.asarray(object_pcd.points)[idx[0]]
        distance = np.linalg.norm(viewpoint_position - nearest_point)
        nearest_distances.append(distance)

        if distance < min_distance:
            # Convert quaternion to rotation matrix
            rotation_matrix = Rotation.from_quat([ox, oy, oz, ow]).as_matrix()
            
            # Get the Z-axis direction (third column of rotation matrix)
            z_axis = rotation_matrix[:, 2]  # Camera Z-axis in base frame
            
            offset_distance = min_distance - distance
            # Move the viewpoint backward along its negative Z-axis
            adjusted_position = viewpoint_position - offset_distance * z_axis
            new_vp = [*adjusted_position, ox, oy, oz, ow]
            adjusted_viewpoints.append(new_vp)
            # print(f"the vp {vp} is adjusted as new_vp: {new_vp}")
        else:
            adjusted_viewpoints.append(vp)

    print(f"nearest_distances before adjusted are: {nearest_distances}")

    nearest_distances_after = []
    for vp in adjusted_viewpoints:
        x, y, z, ox, oy, oz, ow = vp

        # Find the nearest point on the object surface
        viewpoint_position = np.array([x, y, z])
        [_, idx, _] = object_kdtree.search_knn_vector_3d(viewpoint_position, 1)
        nearest_point = np.asarray(object_pcd.points)[idx[0]]
        distance = np.linalg.norm(viewpoint_position - nearest_point)
        nearest_distances_after.append(distance)

    print(f"nearest_distances after adjusted are: {nearest_distances_after}")

    return adjusted_viewpoints

file_path = "8400310XKM42A_new.STL" 
filtered_viewpoint_6Ds_list_base = adjust_viewpoints_if_too_close(file_path, filtered_viewpoint_6Ds_list_base)

print(f"filtered_viewpoint_6Ds_list_base after adjusted is: {filtered_viewpoint_6Ds_list_base}")

def pose_to_transformation_matrix(pose):
    """
    Convert a pose [x, y, z, ox, oy, oz, ow] to a 4x4 transformation matrix.
    """
    x, y, z, ox, oy, oz, ow = pose
    rotation_matrix = Rotation.from_quat([ox, oy, oz, ow]).as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [x, y, z]
    return transformation_matrix

def transform_point_cloud(pcd, transformation_matrix):
    """
    Apply a transformation matrix to a point cloud.
    """
    pcd.transform(transformation_matrix)
    return pcd

def merge_point_clouds(ply_files, poses):
    """
    Load, transform, and merge point clouds from multiple viewpoints.
    
    Args:
        ply_files: List of file paths to the point cloud fragments.
        poses: List of corresponding poses in base frame [x, y, z, ox, oy, oz, ow].
    
    Returns:
        Merged point cloud in the base frame.
    """
    merged_pcd = o3d.geometry.PointCloud()

    for ply_file, pose in zip(ply_files, poses):
        pcd = o3d.io.read_point_cloud(ply_file)  # Load point cloud
        transformation_matrix = pose_to_transformation_matrix(pose)  # Get transformation
        transformed_pcd = transform_point_cloud(pcd, transformation_matrix)  # Transform
        merged_pcd += transformed_pcd  # Merge point clouds

    # Optional: Apply voxel downsampling to reduce noise
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.005)

    return merged_pcd

# # Example usage:
# ply_files = ["ply_00-53-55.ply","ply_00-54-01.ply","ply_00-54-07.ply","ply_00-54-15.ply"
#              ,"ply_00-54-22.ply","ply_00-54-29.ply","ply_00-54-35.ply","ply_00-54-42.ply"
#              ,"ply_00-54-48.ply","ply_00-54-54.ply","ply_00-55-03.ply","ply_00-55-12.ply"]  # List of scanned point cloud fragments

# filtered_viewpoint_6Ds_list_base_reorder = [[0.505679 ,0.372129 ,0.874785 ,0.679306 ,  0.673188, 0.002037 , 0.292160]
#                                                 ,[0.485921 ,0.545722 ,0.797907 ,0.677711 ,  0.615950, -0.054073, 0.397982]
#                                                 ,[0.726444 ,0.519243 ,0.900194 ,0.649884 ,  0.704604, -0.198384, 0.204521]
#                                                 ,[0.423012 ,0.130172 ,0.959621 ,0.673691 ,  0.694444, 0.089513 , 0.236380]
#                                                 ,[0.692647 ,-0.103832, 1.038555, 0.679308,  0.733730, -0.001802, 0.013340]
#                                                 ,[0.941126 ,-0.233288, 1.001513, 0.678050,  0.710926, -0.084144, -0.166594]
#                                                 ,[0.593556 ,-0.566070, 0.974923, 0.630377,  0.729540, 0.251736 ,-0.083812]
#                                                 ,[0.524621 ,-0.613491, 0.951180, 0.614651,  0.732016, 0.288683 ,-0.054935]
#                                                 ,[0.213177 ,-0.674643, 0.644158, 0.390709,  0.695334, 0.571802 ,0.192090]
#                                                 ,[0.109050 ,-0.521054, 0.602719, 0.500031,  0.631879, 0.480066 ,0.346748]
#                                                 ,[1.011881 ,-0.817526, 0.630772, 0.691943,  0.473837, 0.028720 ,-0.543938]
#                                                 ,[0.735649 ,-0.941664, 0.046652, -0.412074, 0.289148, 0.607811 ,0.614129]]


# poses = filtered_viewpoint_6Ds_list_base_reorder  # Corresponding poses in base frame

# # Merge all point clouds into the base frame
# merged_pcd = merge_point_clouds(ply_files, poses)

# # Save or visualize the merged point cloud
# o3d.io.write_point_cloud("merged_point_cloud.ply", merged_pcd)

# merged_pcd.paint_uniform_color([1, 0, 0])

# origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
# o3d.visualization.draw_geometries([merged_pcd,origin_frame])


# # Load object STL (in object frame)
# file_path = "8400310XKM42A_new.STL"
# object_mesh = o3d.io.read_triangle_mesh(file_path)

# # Apply transformation to move STL to base frame
# object_mesh.transform(T_base_object)


# # Visualize both together
# o3d.visualization.draw_geometries([object_mesh, merged_pcd,origin_frame])

# import trimesh
# mesh = trimesh.load(file_path)
# mesh.apply_transform(T_base_object)
# mesh.export("8400310XKM42A_base.STL")