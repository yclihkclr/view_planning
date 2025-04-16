import numpy as np
from scipy.spatial.transform import Rotation as R

# Given transformation from base to object
base_to_object_translation = np.array([1.38 ,-0.074744 ,0.948103])  # (x, y, z)
base_to_object_rpy = np.array([0.007566385, 0.0691947, 1.6306392])  # (roll, pitch, yaw)

# Convert RPY to rotation matrix
base_to_object_rotation = R.from_euler('xyz', base_to_object_rpy).as_matrix()

# Construct homogeneous transformation matrix (4x4)
T_base_object = np.eye(4)
T_base_object[:3, :3] = base_to_object_rotation
T_base_object[:3, 3] = base_to_object_translation

# Define the rotation angle (in degrees) around the object's own y-axis
rotation_angle_deg = -3  # Change this value as needed
rotation_angle_rad = np.radians(rotation_angle_deg)  # Convert to radians

# Get the object's local y-axis (second column of rotation matrix)
y_axis = T_base_object[:3, 1]

# Create a rotation matrix around the local y-axis
rotation_matrix_y = R.from_rotvec(rotation_angle_rad * y_axis).as_matrix()

# Apply the rotation to the object's rotation matrix
T_base_object[:3, :3] = rotation_matrix_y @ T_base_object[:3, :3]


#Apply translation in the rotated object's local frame
local_translation = np.array([-0.018, -0.005, 0.005])  # Translation in object's local frame

# Transform local translation to base frame
world_translation = T_base_object[:3, :3] @ local_translation  # Rotate translation vector

# Update the translation component of T_base_object
T_base_object[:3, 3] += world_translation


# Define the rotation angle (in degrees) around the object's own z-axis
rotation_angle_deg = 1  # Change this value as needed
rotation_angle_rad = np.radians(rotation_angle_deg)  # Convert to radians
# Get the object's local z-axis (second column of rotation matrix)
z_axis = T_base_object[:3, 2]
# Create a rotation matrix around the local z-axis
rotation_matrix_z = R.from_rotvec(rotation_angle_rad * z_axis).as_matrix()
# Apply the rotation to the object's rotation matrix
T_base_object[:3, :3] = rotation_matrix_z @ T_base_object[:3, :3]


#Apply translation in the rotated object's local frame
local_translation = np.array([0.015, 0, 0])  # Translation in object's local frame

# Transform local translation to base frame
world_translation = T_base_object[:3, :3] @ local_translation  # Rotate translation vector

# Update the translation component of T_base_object
T_base_object[:3, 3] += world_translation

# Extract updated translation and convert back to RPY
new_translation = T_base_object[:3, 3]
new_rpy = R.from_matrix(T_base_object[:3, :3]).as_euler('xyz')

# Print the updated values
print("Updated Translation:", new_translation)
print("Updated RPY (roll, pitch, yaw):", new_rpy)