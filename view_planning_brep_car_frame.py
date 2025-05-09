import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
import pdb
import trimesh
import mesh_raycast

def load_cad_model(file_path):
    """
    Load the CAD model and compute normals.
    """
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])
    # pcd = mesh.sample_points_uniformly(number_of_points=1000)
    # o3d.visualization.draw_geometries([pcd])
    return mesh


def compute_viewpoint_surface(mesh, offset_distance, number_of_surface_points, visualize=True):
    """
    Compute the viewpoint surface by offsetting the mesh surface outward.
    Optionally visualize the viewpoint surface.
    """


    # Step 1: Sample points uniformly from the surface
    pcd = mesh.sample_points_poisson_disk(number_of_points=number_of_surface_points, init_factor=5)
    # pcd = mesh.sample_points_uniformly(number_of_points=number_of_surface_points)

    # Step 2: Estimate normals for the point cloud
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    # Step 3: Offset points outward by the fixed distance
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    centroid = np.mean(np.asarray(mesh.vertices), axis=0)
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # # Check if the normal points away from the negtive_z_axis
    # z_axis = origin_frame.get_rotation_matrix_from_xyz((0, 0, 0))[:, 2]
    # negtive_z_axis = -z_axis
    # for i in range(len(normals)):
    #     if np.dot(normals[i], negtive_z_axis) < 0:  
    #         normals[i] = -normals[i]  # Reverse the normal

    # Check if normals point inward and reverse them
    for i in range(len(normals)):
        direction = points[i] - centroid  # Vector from centroid to the point
        if np.dot(normals[i], direction) < 0:  # If normal points inward
            normals[i] = -normals[i]  # Reverse the normal

    # Update the point cloud normals
    pcd.normals = o3d.utility.Vector3dVector(normals)

    offset_points = points + offset_distance * normals

    # visualize
    viewpoint_pcd = o3d.geometry.PointCloud()
    viewpoint_pcd.points = o3d.utility.Vector3dVector(offset_points)

    centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)  # Adjust radius as needed
    centroid_sphere.translate(centroid)  # Move the sphere to the centroid
    centroid_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red color for the centroid spher

    if visualize:
        mesh.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray for the object
        viewpoint_pcd.paint_uniform_color([0.2, 0.8, 0.2])  # Green for the viewpoint surface
        pcd.paint_uniform_color([0.0, 0.0, 1.0]) 
        o3d.visualization.draw_geometries([mesh, origin_frame],
                                          zoom=0.8,
                                    front=[0, 0, 1],
                                    lookat=[0, 1, 0],
                                    up=[1, 0, 0])
        o3d.visualization.draw_geometries([mesh, pcd, centroid_sphere, origin_frame],
                                    zoom=0.8,
                                    front=[0, 0, 1],
                                    lookat=[0, 1, 0],
                                    up=[1, 0, 0])
        o3d.visualization.draw_geometries([mesh, pcd, viewpoint_pcd, centroid_sphere, origin_frame],
                                    zoom=0.8,
                                    front=[0, 0, 1],
                                    lookat=[0, 1, 0],
                                    up=[1, 0, 0])
    surface_pcd = pcd 
    return viewpoint_pcd, surface_pcd


def discretize_viewpoint_surface(mesh, viewpoint_pcd, voxel_size, visualize=True):
    """
    Discretize the viewpoint surface into candidate viewpoints using voxelization.
    Optionally visualize the discretized surface.
    """
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(viewpoint_pcd, voxel_size=voxel_size)
    
    # Get the voxel grid origin and step size
    origin = voxel_grid.origin
    voxel_size = voxel_grid.voxel_size

    # Extract all voxel center coordinates
    candidate_viewpoints = []
    for voxel in voxel_grid.get_voxels():
        grid_index = np.array(voxel.grid_index, dtype=float)
        center_coord = origin + grid_index * voxel_size
        candidate_viewpoints.append(center_coord)
    
    if visualize:
        # Convert candidate viewpoints to a point cloud
        candidate_pcd = o3d.geometry.PointCloud()
        candidate_pcd.points = o3d.utility.Vector3dVector(candidate_viewpoints)
        
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

        # Visualization
        mesh.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray for the object
        viewpoint_pcd.paint_uniform_color([0.2, 0.8, 0.2])  # Green for the viewpoint surface
        candidate_pcd.paint_uniform_color([1, 0, 0])  # Red for the candidate viewpoints
        o3d.visualization.draw_geometries([mesh, candidate_pcd, origin_frame],
                                          zoom=0.8,
                                    front=[0, 0, 1],
                                    lookat=[0, 1, 0],
                                    up=[1, 0, 0])
    return np.array(candidate_viewpoints)

def compute_visibility_brep(file_path, surface_pcd, candidate_viewpoints, scanning_specs):
    """
    Compute visibility of surface points from candidate viewpoints using ray-casting, FOV checks, and result validation.

    Parameters:
        mesh: Open3D triangle mesh object.
        surface_pcd: Open3D point cloud representing the surface.
        candidate_viewpoints: List of candidate viewpoint coordinates.
        scanning_specs: Dictionary containing the scanning model specifications:
            - "min_dist": Minimum scanning distance.
            - "max_dist": Maximum scanning distance.
            - "area_func": Function to compute scanning area at a given distance.

    Returns:
        visible_dict: Dictionary mapping candidate viewpoints to visible surface points and angle contributions.
    """
    mesh = trimesh.load_mesh(file_path)
    triangles = mesh.vertices[mesh.faces]
    triangles = np.array(triangles, dtype='f4')
    # # Convert Open3D mesh to arrays of vertices and faces
    # vertices = np.asarray(mesh.vertices)
    # faces = np.asarray(mesh.triangles)

    # Prepare surface points and normals
    surface_points = np.asarray(surface_pcd.points)
    surface_normals = np.asarray(surface_pcd.normals)

    # Scanning specifications
    min_dist = scanning_specs["min_dist"]
    max_dist = scanning_specs["max_dist"]
    area_func = scanning_specs["area_func"]

    # Initialize the visible dictionary
    visible_dict = {i: [] for i in range(len(candidate_viewpoints))}
    print("We have total vp %d, total sp %d"%(len(candidate_viewpoints) , len(surface_points)))
    for i, vp in enumerate(candidate_viewpoints):
        for j, sp in enumerate(surface_points):
            direction = sp - vp
            distance = np.linalg.norm(direction)
            ray_direction = direction / distance

            # Check if the surface point is within the scanning range
            if not (min_dist <= distance <= max_dist):
                continue


            # Check if the surface point is within the FOV
            if not is_point_in_fov(vp, sp, -surface_normals[j], ray_direction, distance, area_func):
                continue

            # Perform ray-casting using mesh_raycast
            result = mesh_raycast.raycast(source=vp, direction=ray_direction, mesh=triangles)

            
            # Validate ray-casting result
            if result is None:
                continue
            first_result = min(result, key=lambda x: x['distance'])
            # pdb.set_trace()
            hit_point = np.array(first_result['point'])

            # Check for invalid results (e.g., inf, nan, or unrealistic distances)
            if (
                not np.isfinite(hit_point).all() 
            ):
                continue

            # Check if the ray hits the current surface point
            if np.allclose(hit_point, sp, atol=1e-2):
                # print("sp is %s, the hit point is %s"%(sp,hit_point) )
                # Compute angle contribution
                cos_theta = np.dot(-ray_direction, surface_normals[j])
                angle_contribution = max(0, cos_theta - np.cos(np.radians(30)))  # 30 degree threshold

                # Add to visible_dict
                visible_dict[i].append((j, angle_contribution))
            print("Current visibility checking vp %d, sp %d"%(i,j))
            
    return visible_dict

def is_point_in_fov(vp, sp, negtive_surface_normal, connecting_direction, connecting_distance, area_func, up_vector=np.array([0, 0, 1])):
    """
    Check if a surface point is within the FOV of the camera at a given viewpoint.

    Parameters:
        vp: Camera position (viewpoint).
        sp: Surface point position.
        direction: Camera viewing direction (unit vector).
        distance: Distance from the camera to the surface point.
        area_func: Function to compute FOV dimensions at a given distance.
        up_vector: Up vector defining the camera's orientation.

    Returns:
        True if the point is within the FOV, False otherwise.
    """
    cos_theta= np.dot(negtive_surface_normal,connecting_direction)
    if cos_theta< 0:
        # Compute the camera's local coordinate system
        camera_z = -negtive_surface_normal  # Viewing direction
        camera_x = np.cross(up_vector, camera_z)
        if np.linalg.norm(camera_x) < 1e-5:
            raise ValueError("Up vector is parallel to viewing direction.")
        camera_x /= np.linalg.norm(camera_x)  # Normalize
        camera_y = np.cross(camera_z, camera_x)

        # Transform surface point to camera's local frame
        local_point = sp - vp
        x_local = np.dot(local_point, camera_x)  # Project onto camera_x
        y_local = np.dot(local_point, camera_y)  # Project onto camera_y
        z_local = np.dot(local_point, camera_z)  # Project onto camera_z

        # Ensure the point is in front of the camera
        if z_local <= 0:
            return False
        fov_distance = -cos_theta * connecting_distance
    
    elif cos_theta> 0:
        # Compute the camera's local coordinate system
        camera_z = negtive_surface_normal  # Viewing direction
        camera_x = np.cross(up_vector, camera_z)
        if np.linalg.norm(camera_x) < 1e-5:
            raise ValueError("Up vector is parallel to viewing direction.")
        camera_x /= np.linalg.norm(camera_x)  # Normalize
        camera_y = np.cross(camera_z, camera_x)

        # Transform surface point to camera's local frame
        local_point = sp - vp
        x_local = np.dot(local_point, camera_x)  # Project onto camera_x
        y_local = np.dot(local_point, camera_y)  # Project onto camera_y
        z_local = np.dot(local_point, camera_z)  # Project onto camera_z

        # Ensure the point is in front of the camera
        if z_local <= 0:
            return False
        fov_distance = cos_theta * connecting_distance
        
    elif cos_theta== 0:
        return True

    # Check if the point is within the FOV
    scanning_area = area_func(fov_distance)
    half_width, half_height = scanning_area[0]*0.7 , scanning_area[1]*0.7 

    return -half_width <= x_local <= half_width and -half_height <= y_local <= half_height

def compute_visibility_and_value_brep(file_path, surface_pcd, candidate_viewpoints, scanning_specs, w_coverage=0.7, w_angle=0.3):
    """
    Compute visibility using B-Rep and assign normalized values to viewpoints.

    Parameters:
        mesh: The CAD mesh object.
        surface_pcd: Downsampled surface points as a point cloud.
        candidate_viewpoints: List of candidate viewpoints.
        w_coverage: Weight for coverage contribution.
        w_angle: Weight for angle alignment.

    Returns:
        visible_dict: Dictionary mapping candidate viewpoints to visible points.
        viewpoint_values: Array of normalized scores for each viewpoint.
    """
    visible_dict = compute_visibility_brep(file_path, surface_pcd, candidate_viewpoints,scanning_specs)

    # Initialize contributions
    surface_points = np.asarray(surface_pcd.points)
    coverage_contributions = np.zeros(len(candidate_viewpoints))
    view_angle_contributions = np.zeros(len(candidate_viewpoints))

    # Process visible_dict
    for i, visible_points in visible_dict.items():
        if isinstance(visible_points, dict):  # Ensure it handles dictionary format
            angle_scores = [value for _, value in visible_points.items()]
        else:  # Handle as list of tuples (point, score)
            angle_scores = [value for _, value in visible_points]

        # Coverage contribution
        coverage_contributions[i] = len(visible_points) / len(surface_points)  # Fraction of visible points

        # Angle contribution
        view_angle_contributions[i] = np.mean(angle_scores) if angle_scores else 0

    # Normalize contributions
    if np.max(coverage_contributions) > 0:
        coverage_contributions /= np.max(coverage_contributions)  # Normalize to [0, 1]
    if np.max(view_angle_contributions) > 0:
        view_angle_contributions /= np.max(view_angle_contributions)  # Normalize to [0, 1]

    # Final viewpoint values
    viewpoint_values = (
        w_coverage * coverage_contributions +
        w_angle * view_angle_contributions
    )

    return visible_dict, viewpoint_values

def visualize_visibility_brep(mesh, surface_pcd, candidate_viewpoints, visible_dict,viewpoint_values):
    """
    Visualize visibility, candidate viewpoints, and values with enhanced contrast.
    """
    # Paint the mesh and surface points
    mesh.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray for the mesh
    surface_pcd.paint_uniform_color([0, 0, 1])  # Blue for the surface points

    # Normalize viewpoint values for enhanced contrast
    normalized_viewpoint_values = (viewpoint_values - np.min(viewpoint_values)) / (
        np.max(viewpoint_values) - np.min(viewpoint_values)
    )

    # Visualize candidate viewpoints
    viewpoint_spheres = []
    for i, vp in enumerate(candidate_viewpoints):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(vp)
        # Color the sphere based on normalized viewpoint value (intense red for high, blue for low)
        intensity = normalized_viewpoint_values[i]
        sphere.paint_uniform_color([intensity, 0, 1 - intensity])  # Red to blue gradient
        viewpoint_spheres.append(sphere)

    # Create arrows from viewpoints to visible surface points
    lines = []
    colors = []

    # Find min and max angle contribution for normalization
    min_value = min(value for _, points in visible_dict.items() for _, value in points)
    max_value = max(value for _, points in visible_dict.items() for _, value in points)

    for vp_index, visible_points in visible_dict.items():
        vp = candidate_viewpoints[vp_index]
        for point_index, value in visible_points:
            sp = np.asarray(surface_pcd.points)[point_index]
            lines.append([vp, sp])

            # Normalize angle values to range [0, 1]
            if max_value > min_value:
                normalized_value = (value - min_value) / (max_value - min_value)
            else:
                normalized_value = 0.5  # Default to midpoint if all values are identical

            # Map to yellow (high) to green (low)
            color = [(1 - normalized_value), 1, 0]  # Yellow to green
            colors.append(color)

    # Prepare line set for arrows
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack(lines))
    line_set.lines = o3d.utility.Vector2iVector([[i * 2, i * 2 + 1] for i in range(len(lines))])
    line_set.colors = o3d.utility.Vector3dVector(colors)

    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # Visualize everything
    o3d.visualization.draw_geometries(
        [mesh, surface_pcd, line_set, origin_frame] + viewpoint_spheres,
                                    zoom=0.8,
                                    front=[0, 0, 1],
                                    lookat=[0, 1, 0],
                                    up=[1, 0, 0])
    

def compute_mean_direction(viewpoint, visible_points, surface_points):
    """
    Compute the mean direction vector from the viewpoint to its visible points.

    Parameters:
        viewpoint: 6D viewpoint (position + orientation).
        visible_points: List of visible points (indices and values).
        surface_points: Array of 3D surface points.

    Returns:
        mean_direction: Normalized mean direction vector.
    """
    # Extract only the 3D position from the 6D viewpoint
    position = viewpoint[:3]

    # Compute directions
    directions = np.array([surface_points[pt_idx] - position for pt_idx, _ in visible_points])

    # Normalize each direction vector
    normalized_directions = np.array([d / np.linalg.norm(d) for d in directions if np.linalg.norm(d) > 1e-6])

    # Compute the mean direction
    mean_direction = np.mean(normalized_directions, axis=0)

    # Normalize the mean direction to ensure it's a unit vector
    if np.linalg.norm(mean_direction) > 1e-6:
        mean_direction /= np.linalg.norm(mean_direction)
    else:
        raise ValueError("Mean direction cannot be computed due to insufficient data.")

    return mean_direction

def rotation_matrix_from_vectors(vec1, vec2):
    # Normalize input vectors
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    # Compute the rotation axis (cross product)
    axis = np.cross(vec1, vec2)
    axis_len = np.linalg.norm(axis)
    axis = axis / axis_len if axis_len > 1e-6 else axis  # Avoid division by zero

    # Compute the angle of rotation (dot product)
    angle = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))

    # Create the rotation matrix using Rodrigues' formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    return rotation_matrix


def orient_viewpoint(viewpoint, visible_points, surface_points):
    # Compute the mean direction from the viewpoint to the visible points
    mean_direction = compute_mean_direction(viewpoint, visible_points, surface_points)

    # The viewpoint's initial z-axis is assumed to be [0, 0, 1]
    initial_z_axis = np.array([0, 0, 1])

    # Compute the rotation matrix to align the z-axis with the mean direction
    rotation_matrix = rotation_matrix_from_vectors(initial_z_axis, mean_direction)

    # Apply the rotation to the viewpoint's orientation (this can be a 6D transformation)
    # Assuming viewpoint has position [x, y, z] and orientation in rotation matrix form
    new_orientation = rotation_matrix @ np.array([0, 0, 1])  # Apply rotation to the z-axis

    # Update the viewpoint with new orientation (You can store the rotation matrix or Euler angles)
    return new_orientation

def compute_distance_between_viewpoints(vp1, vp2):
    """ Compute Euclidean distance between two viewpoints """
    return np.linalg.norm(vp1[:3] - vp2[:3])

def compute_min_distance_to_selected_viewpoints(vp, selected_viewpoints, candidate_viewpoints, min_distance):
    """ Compute the minimum Euclidean distance from the new viewpoint to the selected viewpoints """
    if len(selected_viewpoints) == 0:
        m_distance = min_distance
    else:
        distances = [compute_distance_between_viewpoints(candidate_viewpoints[vp], candidate_viewpoints[selected_vp]) for selected_vp in selected_viewpoints]
        m_distance = min(distances)
    return m_distance

def optimize_viewpoints_with_values(visible_dict, viewpoint_values, surface_pcd, candidate_viewpoints, mesh, scanning_specs, coverage_threshold=1.0, min_distance=0.1):
    surface_points = np.asarray(surface_pcd.points)
    num_surface_points = len(surface_points)
    covered_points = set()
    selected_viewpoints = []
    uncovered_points_set = set()

    viewpoint_to_points = {vp: {pt for pt, _ in visible_points} for vp, visible_points in visible_dict.items()}

    while len(covered_points) < num_surface_points * coverage_threshold:
        best_viewpoint = None
        best_score = -np.inf

        for vp, points in viewpoint_to_points.items():
            uncovered_points = points - covered_points
            coverage_contribution = len(uncovered_points)
            if coverage_contribution > 0:
                # Calculate score based on coverage and viewpoint value
                score = coverage_contribution
                score += viewpoint_values[vp]*0.5

                # Redundancy penalty: subtract penalty for covering points already covered
                repetitive_points = points.intersection(covered_points)
                redundancy_penalty = sum(1 for pt in repetitive_points if pt in covered_points)
                score -= redundancy_penalty * 0.05

                # Distance-based penalty: penalize viewpoints too close to selected viewpoints
                min_distance_to_selected = compute_min_distance_to_selected_viewpoints(vp, selected_viewpoints, candidate_viewpoints, min_distance)
                if min_distance_to_selected < min_distance:
                    distance_penalty = (min_distance - min_distance_to_selected) * 10  # High penalty if too close
                    score -= distance_penalty

                if score > best_score:
                    best_viewpoint = vp
                    best_score = score

        if best_viewpoint is None:
            print("Warning: Full coverage is not achievable. Returning partial solution.")
            print("The final coverage rate is:", len(covered_points)/num_surface_points)
            break

        # Add the selected viewpoint
        selected_viewpoints.append(best_viewpoint)
        covered_points.update(viewpoint_to_points[best_viewpoint])
        del viewpoint_to_points[best_viewpoint]

        # Now orient the selected viewpoint
        new_orientation = orient_viewpoint(candidate_viewpoints[best_viewpoint], visible_dict[best_viewpoint], surface_points)

        # Ensure candidate_viewpoints can handle 6D data
        if isinstance(candidate_viewpoints[0], np.ndarray) and candidate_viewpoints[0].shape == (3,):
            # Initialize a new structure to store 6D viewpoints
            candidate_viewpoints = [np.hstack([vp, [0, 0, 0]]) for vp in candidate_viewpoints]

        # Update the best viewpoint with the new orientation
        candidate_viewpoints[best_viewpoint][3:] = new_orientation

    total_points = set(range(len(surface_points)))
    uncovered_points_set = total_points - covered_points
    # Visualize selected viewpoints and surface points
    viewpoint_6Ds = visualize_selected_viewpoints(mesh, surface_pcd, selected_viewpoints, candidate_viewpoints, uncovered_points_set, scanning_specs)
    
    print("The final coverage rate is:", len(covered_points)/num_surface_points)

    return selected_viewpoints, viewpoint_6Ds
   

def visualize_selected_viewpoints(mesh, surface_pcd, selected_viewpoints, candidate_viewpoints, uncovered_points_set, scanning_specs):
    """
    Visualize the final selected viewpoints along with the mesh, surface points, uncovered points, and camera frustums.

    Parameters:
        mesh: The object mesh.
        surface_pcd: The surface points of the object.
        selected_viewpoints: List of indices of selected viewpoints.
        candidate_viewpoints: List of 6D candidate viewpoints (position and orientation).
        uncovered_points_set: Set of indices of uncovered surface points.
        scanning_specs: Dictionary containing camera FOV specifications.
    """

    # Paint surface points blue
    surface_pcd.paint_uniform_color([0, 0, 1])

    # Visualize candidate viewpoints
    viewpoint_spheres = []
    for i, vp in enumerate(candidate_viewpoints):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(vp[:3])
        sphere.paint_uniform_color([1, 0, 0])  # Red for candidate viewpoints
        viewpoint_spheres.append(sphere)

    # Create coordinate frames and frustums for selected viewpoints
    viewpoint_frames = []
    viewpoint_6Ds = []
    frustums = []
    for vp_idx in selected_viewpoints:
        vp_position = candidate_viewpoints[vp_idx][:3]
        z_axis = candidate_viewpoints[vp_idx][3:]  # Extract orientation (Z-axis direction vector)

        # Normalize the Z-axis direction
        z_axis = np.array(z_axis) / np.linalg.norm(z_axis)

        # Default X-axis and compute Y-axis via cross-product
        default_x_axis = np.array([1, 0, 0])
        y_axis = np.cross(z_axis, default_x_axis)
        if np.linalg.norm(y_axis) < 1e-6:  # Handle case when z_axis == default_x_axis
            default_x_axis = np.array([0, 1, 0])
            y_axis = np.cross(z_axis, default_x_axis)

        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)

        # Construct rotation matrix
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

        # Create and rotate the coordinate frame for the viewpoint
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06, origin=vp_position)
        frame.rotate(rotation_matrix, center=vp_position)
        viewpoint_frames.append(frame)
        viewpoint_6Ds.append((vp_position, rotation_matrix))

        # Create the camera FOV frustum
        min_dist = scanning_specs["min_dist"]
        max_dist = scanning_specs["max_dist"]
        area_func = scanning_specs["area_func"]

        # Compute frustum corners at min and max distances in the local frame
        for dist in [min_dist, max_dist]:
            half_width, half_height = np.array(area_func(dist)) / 2
            local_corners = np.array([
                [-half_width, -half_height, dist],
                [half_width, -half_height, dist],
                [half_width, half_height, dist],
                [-half_width, half_height, dist]
            ])
            # Rotate and translate the corners to the world frame
            world_corners = (rotation_matrix @ local_corners.T).T + vp_position

            if dist == min_dist:
                min_corners = world_corners
            else:
                max_corners = world_corners

        # Create the frustum lines
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Base rectangle (far plane)
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top rectangle (near plane)
            [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting lines
        ]

        # Combine min and max corners
        frustum_points = np.vstack((min_corners, max_corners))

        # Generate a unique color for each frustum
        color = np.random.rand(3)

        # Create the frustum visualization
        frustum = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(frustum_points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        frustum.paint_uniform_color(color)
        frustums.append(frustum)

    # Origin frame for reference
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # Visualize uncovered points as red spheres
    uncovered_points_spheres = []
    for point_idx in uncovered_points_set:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(surface_pcd.points[point_idx])
        sphere.paint_uniform_color([1, 0, 0])  # Red color for uncovered points
        uncovered_points_spheres.append(sphere)

    # Combine all the objects for visualization
    visualization_objects = [mesh, surface_pcd, origin_frame] + viewpoint_spheres + viewpoint_frames
    o3d.visualization.draw_geometries(visualization_objects, zoom=0.8,
                                    front=[0, 0, 1],
                                    lookat=[0, 1, 0],
                                    up=[1, 0, 0])
    

    # Combine objects with uncovered points for final visualization
    visualization_objects_with_uncovered_points = (
        [mesh, surface_pcd, origin_frame] + viewpoint_frames + frustums + uncovered_points_spheres
    )
    o3d.visualization.draw_geometries(visualization_objects_with_uncovered_points, zoom=0.8,
                                    front=[0, 0, 1],
                                    lookat=[0, 1, 0],
                                    up=[1, 0, 0])
    

    return viewpoint_6Ds

def main():
    # file_path = "8400310XKM42A_new.STL"  # Replace with your CAD model path
    file_path = "car_frame.STL"

    # offset_distance = 0.442  # Distance from the object surface, which should be the sweet point of the camera 0.442
    # offset_distance = 0.3
    offset_distance = 0.4

    # number_of_surface_points = 500 # Number of downsampled surface points
    number_of_surface_points = 1500
    
    voxel_size = 0.02      # Resolution of viewpoint discretization

    # the FOV model of the camera based on distance, the given model is the Photoneo PhoXi 3D Scanner S
    # scanning_specs = {
    #     "min_dist": 0.384,  # Minimum scanning distance in meters
    #     "max_dist": 0.520,  # Maximum scanning distance in meters
    #     "area_func": lambda d: (0.343 + (0.382 - 0.343) * (d - 0.384) / (0.520 - 0.384),
    #                             0.237 + (0.319 - 0.237) * (d - 0.384) / (0.520 - 0.384))
    # }
    # the FOV model of the camera based on distance, the given model is the UHP
    # scanning_specs = {
    #     "min_dist": 0.28,  # Minimum scanning distance in meters
    #     "max_dist": 0.32,  # Maximum scanning distance in meters
    #     "area_func": lambda d: (0.135 + (0.15 - 0.142) * (d - 0.3) / (0.32 - 0.3),
    #                             0.095 + (0.1 - 0.095) * (d - 0.3) / (0.32 - 0.3))
    # }
    # A10400
    scanning_specs = {
        "min_dist": 0.25,  # Minimum scanning distance in meters
        "max_dist": 0.60,  # Maximum scanning distance in meters
        "area_func": lambda d: (0.1826137 + (0.4255416 - 0.1826137) * (d - 0.25) / (0.6 - 0.25),
                                0.1399103 + (0.3559081 - 0.1399103) * (d - 0.25) / (0.6 - 0.25))
    }

    # Load CAD model
    mesh = load_cad_model(file_path)

    # Compute viewpoint surface
    viewpoint_pcd, surface_pcd = compute_viewpoint_surface(mesh, offset_distance, number_of_surface_points)

    # Discretize viewpoint surface
    candidate_viewpoints = discretize_viewpoint_surface(mesh, viewpoint_pcd, voxel_size)

    # Compute visibility to get dictionary of viewpoint to visible point and get scores for the viewpoint
    visible_dict, viewpoint_values = compute_visibility_and_value_brep(file_path, surface_pcd, candidate_viewpoints, scanning_specs)
    # Visualize visibility
    visualize_visibility_brep(mesh, surface_pcd, candidate_viewpoints, visible_dict, viewpoint_values)
    

    #Optimize viewpoints and get the plan
    selected_viewpoints_idx, viewpoint_6Ds = optimize_viewpoints_with_values(
                                visible_dict=visible_dict,
                                viewpoint_values=viewpoint_values,
                                surface_pcd=surface_pcd,
                                candidate_viewpoints=candidate_viewpoints,
                                mesh=mesh,
                                scanning_specs= scanning_specs)

    print( "The number of planned viewpoints are : ",len(selected_viewpoints_idx))
    print(f"Selected Viewpoints idx: {selected_viewpoints_idx}")
    print(f"Selected Viewpoints 6d in object frame are: {viewpoint_6Ds}")
    # print(f"Selected Viewpoints frames: {viewpoint_6Ds}")



if __name__ == "__main__":
    main()