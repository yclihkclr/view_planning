import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R

def visualize_selected_viewpoints(object_mesh, viewpoint_6Ds_list_base, frustum_size=0.1):
    """
    Visualize selected viewpoints as coordinate frames along with the object mesh.
    
    Parameters:
        object_mesh: The 3D mesh of the object (in base frame).
        viewpoint_6Ds_list_base: List of viewpoint poses in base frame as [x, y, z, ox, oy, oz, ow].
        frustum_size: Scaling factor for the camera frustum visualization.
    """

    # Load object mesh if it's a file path
    if isinstance(object_mesh, str):
        object_mesh = o3d.io.read_triangle_mesh(object_mesh)
    
    visualization_objects = [object_mesh]

    # Iterate over each viewpoint pose
    for vp in viewpoint_6Ds_list_base:
        x, y, z, ox, oy, oz, ow = vp

        # Convert quaternion to rotation matrix
        rotation_matrix = R.from_quat([ox, oy, oz, ow]).as_matrix()

        # Create a coordinate frame for the viewpoint
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[x, y, z])
        frame.rotate(rotation_matrix, center=[x, y, z])
        visualization_objects.append(frame)

    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    visualization_objects.append(origin_frame)

    # Create a custom visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window("vp_visualize_object")  # Set custom window title

    for geom in visualization_objects:
        vis.add_geometry(geom)

    vis.run()
    vis.destroy_window()

    # # Visualize everything together
    # o3d.visualization.draw_geometries(visualization_objects)

file_path = "../data/8400310XKM42A_new.STL"  # Replace with your CAD model path

# Load object STL (in object frame)
object_mesh = o3d.io.read_triangle_mesh(file_path)

viewpoint_6Ds_list_object = [[0.01926945, 0.63325641, 0.01230041, 0.9991732992531206, 4.9463342879549545e-05, -0.0012162410598199492, -0.04063540784458962], [-0.13073055, 1.03325641, -0.83769959, 0.3410797939561727, -0.0738376576650309, -0.026885174854899663, 0.9367442350174235], [0.56926945, 0.83325641, -0.13769959, 0.9376880615873122, -0.06199607037478232, -0.2540012451745129, 0.22886885734959858], [-0.53073055, 0.48325641, -0.18769959, 0.914122103771477, -0.08006106323314524, 0.3296873376918498, -0.22198483037371589], [0.01926945, 0.88325641, -0.93769959, 0.14013828845771126, -0.182341439103902, -0.02626634008069017, 0.9728427103455248], [-0.23073055, 0.18325641, -0.58769959, -0.626321990833703, 0.06271321465745595, -0.05065699210489718, 0.7753848629269977], [-0.48073055, 0.38325641, -0.83769959, -0.4364930008173801, 0.16166794579891766, -0.08005912857400861, 0.8814351203977336], [0.11926945, 0.93325641, -0.93769959, 0.16052052742168485, -0.04990452030249522, -0.008126613078862007, 0.9857366064469294], [-0.83073055, 0.48325641, -0.73769959, -0.3928392332635902, 0.5670053737977409, -0.3520920286263152, 0.6326242536222944], [-0.83073055, 0.48325641, -0.63769959, -0.49477439554811214, 0.42677567070772154, -0.3047129266271467, 0.6929724790699079], [-0.28073055, 0.48325641, 0.01230041, 0.9795166814984156, -0.020478392930872922, 0.14006891102133606, -0.1432075635383732], [-0.13073055, 0.98325641, -0.08769959, 0.9401494899927937, 0.03335163223554056, 0.0964417372674789, 0.325123970827046], [0.76926945, 0.63325641, -0.13769959, 0.9276858888190307, -0.019374866192766828, -0.36967492549089026, 0.04862052765408861], [0.26926945, 1.03325641, -0.08769959, 0.9419172162144751, -0.04022455660461276, -0.12211761039831509, 0.3102599427501742], [-0.48073055, 0.83325641, -0.18769959, 0.9218414759842898, 0.06107046485024909, 0.3465343940102557, 0.16245801071669458], [-0.03073055, 0.73325641, 0.06230041, 0.9997787716448732, -7.218164987635354e-05, -0.003478926309581507, 0.020743664811466504], [-0.38073055, 0.83325641, -0.08769959, 0.9667542620167563, 0.028538428639528837, 0.22148267512893882, 0.12456797173010581], [0.06926945, 0.88325641, -0.03769959, 0.9757283094478677, -0.010694201332938642, -0.04894847979734855, 0.21317585821425766]]
visualize_selected_viewpoints(object_mesh, viewpoint_6Ds_list_object)
