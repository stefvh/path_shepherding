from enum import Enum
import numpy as np


def calculate_relative_positions(robot_poses: np.ndarray, herd_poses: np.ndarray):
    herd_mean_position = np.mean(herd_poses[:, :2], axis=0)
    herd_mean_orientation = np.arctan2(
        np.mean(np.sin(herd_poses[:, 2])), np.mean(np.cos(herd_poses[:, 2]))
    )

    relative_positions = rotate_positions(robot_poses[:, :2], herd_mean_position, herd_mean_orientation)

    return relative_positions

def rotate_positions(positions, pivot, angle):
    # Translate positions to pivot
    translated_positions = positions - pivot

    # Rotate positions
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    rotated_positions = np.dot(translated_positions, rotation_matrix)

    return rotated_positions

def label_robot_positions(robot_poses: np.ndarray, herd_poses: np.ndarray):
    relative_positions = calculate_relative_positions(robot_poses, herd_poses)
    headings = np.arctan2(relative_positions[:, 1], relative_positions[:, 0])

    labels = []
    for heading in headings:
        angle_degrees = np.degrees(heading)

        # Determine quadrant based on relative position and orientation
        if 0 <= angle_degrees < 45 or -45 <= angle_degrees < 0:
            labels.append("FRONT")
        elif 135 <= angle_degrees < 180 or -180 <= angle_degrees < -135:
            labels.append("BACK")
        elif 45 <= angle_degrees < 135:
            labels.append("LEFT")
        else:
            labels.append("RIGHT")

    return labels
