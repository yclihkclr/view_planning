import open3d as o3d
import numpy as np

def save_camera_params(vis):
    """
    Extract and print camera parameters after manual adjustment.
    """
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()

    # Save extrinsic matrix (camera pose)
    np.savetxt("camera_extrinsic.txt", cam_params.extrinsic)

    print("Camera parameters saved!")

def visualize_and_save(mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)

    vis.run()  # Adjust the view manually
    save_camera_params(vis)  # Save the camera parameters
    vis.destroy_window()

# Load your mesh
mesh = o3d.io.read_triangle_mesh("8400310XKM42A_new.STL")

# Run visualization and save camera parameters
visualize_and_save(mesh)

def load_camera_params(vis):
    """
    Load and apply saved camera parameters.
    """
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()

    # Load saved extrinsic matrix
    cam_params.extrinsic = np.loadtxt("camera_extrinsic.txt")

    ctr.convert_from_pinhole_camera_parameters(cam_params)
    print("Camera parameters applied!")

def visualize_with_saved_view(mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)

    vis.run()  # Initialize first

    load_camera_params(vis)  # Apply saved parameters

    vis.run()  # Re-run to see the correct view
    vis.destroy_window()

# Run visualization with the fixed camera view
visualize_with_saved_view(mesh)