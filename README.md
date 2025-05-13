# Viewpoint Planning for RoboScan 3D Scanning and Object Pose Initial Estimation
This repository provides Python scripts for 3D scanning and pose estimation of objects using point clouds and mesh models. The tools support tasks like initial pose estimation, viewpoint planning, result transformation, visualization, and pose calibration, ideal for the RoboScan applications.

## Dependencies
To run the scripts, you need:

- Python 3.6+
- open3d
- numpy
- scipy
- trimesh
- matplotlib
- mesh_raycast (custom module)

## Installation
### Prerequisites
Ensure you have a C++ compiler and development tools installed, as mesh_raycast requires compilation:

```
sudo apt update
sudo apt install -y build-essential python3-dev g++
```

### Install Dependencies
Install all dependencies using the provided requirements.txt file::

```
python -m pip install -r requirements.txt 
```

This installs open3d, numpy, scipy, trimesh, matplotlib, and mesh_raycast from its GitHub repository.
If the mesh_raycast git dependency fails (e.g., due to network issues or repository access), install it manually:

```
git clone https://github.com/szzhu-hkclr/python-mesh-raycast
cd python-mesh-raycast
python -m pip install .
cd ..
python -m pip install open3d numpy scipy trimesh matplotlib
```

## Usage
Each script can be run with Python. Ensure input files (e.g., STL models, point cloud files) are correctly specified in the scripts.

```
python view_planning_brep.py data/car_frame.json
```

- Plans optimal camera viewpoints for 3D scanning of a CAD model, maximizing surface coverage while considering visibility and scanner constraints.
- Inputs: STL file path, scanner specs (e.g., FOV, distance).
- Outputs: Selected viewpoint indices, 6D poses, visualization.



```
python initial_pose_estimation.py
```

- Estimates an object’s 3D pose by aligning point cloud fragments (PLY files) with a reference mesh (STL file) using RANSAC and ICP algorithms.
- Inputs: Point cloud files, camera poses, mesh file path.
- Outputs: Transformation matrix, visualization (if enabled).



```
python result_transform.py
```

- Transforms viewpoints from the object frame to the base frame, filters them (e.g., by angle), adjusts for distance, and optionally merges point clouds.
- Inputs: Viewpoint poses, transformation parameters, STL file.
- Outputs: Transformed viewpoints, registered point cloud (optional).



```
python vp_visualize_object.py
```

- Visualizes the object mesh and selected viewpoints as coordinate frames in the object frame.
- Inputs: STL file or mesh, viewpoint poses.
- Outputs: 3D visualization.



```
python vp_visualize_base.py
```

- Visualizes the object mesh and viewpoints in the base frame, showing their spatial arrangement.
- Inputs: STL file or mesh, viewpoint poses, transformation parameters.
- Outputs: 3D visualization.



```
python set_view.py
```

- Saves camera parameters after manual adjustment and reloads them for consistent mesh visualization.
- Inputs: STL file.Outputs: Camera parameters file, visualization.



```
python part_pose_calibration.py
```
- Calibrates an object’s pose by applying local rotations and translations in its coordinate frame.
- Inputs: Initial translation, RPY angles, rotation/translation adjustments.
- Outputs: Updated translation, RPY angles.



## Trouble Shooting
- For detailed configuration (e.g., file paths, parameters), refer to the comments and code within each script.
- Ensure the STL file path (e.g., data/8400310XKM42A_new.STL) in scripts like view_planning_brep.py is valid.
- Scripts use Open3D for visualization; ensure your system supports 3D rendering.
- If using a ROS environment, test without sourcing ROS setup scripts to avoid conflicts.
- Permission Issues: If you see Defaulting to user installation, ensure write permissions for the Python site-packages directory in the Docker container, or use 

```
python -m pip install . --user
```

- Network issue during pip install, try https://pypi.tuna.tsinghua.edu.cn/simple, i.e.

```
python -m pip install trimesh -i https://pypi.tuna.tsinghua.edu.cn/simple
```

