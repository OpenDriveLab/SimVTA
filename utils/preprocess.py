import torch
def radians_to_degrees(radians):
    return radians * 180.0 / torch.pi

def extract_yaw_from_rotation_matrices(rot_matrices):
    return torch.atan2(rot_matrices[..., 1, 0], rot_matrices[..., 0, 0])

def get_rel_pose(poses):
    poses = torch.cat([poses[:, 0:1], poses], dim=1)
    rot_A = poses[:, :-1, :3, :3]
    rot_B = poses[:, 1:, :3, :3]
    trans_A = poses[:, :-1, :3, 3:]
    trans_B = poses[:, 1:, :3, 3:]
    rel_pose_rot = torch.linalg.inv(rot_A) @ rot_B
    rel_pose_trans = torch.linalg.inv(rot_A) @ (trans_B - trans_A)
    rel_poses = rel_pose_trans[..., :2, 0]
    rel_yaws = radians_to_degrees(extract_yaw_from_rotation_matrices(rel_pose_rot[..., :3, :3])).unsqueeze(-1)
    return (rel_poses, rel_yaws)
