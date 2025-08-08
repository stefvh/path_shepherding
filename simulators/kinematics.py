import numpy as np

from utils import maths

from algorithms.base import handle_collision_avoidance_hard


class Kinematics:
    def __init__(
        self,
        simulation_time_step,
        angular_noise,
        max_linear_velocity,
        max_angular_velocity,
    ) -> None:
        self.simulation_time_step = simulation_time_step
        self.angular_noise = angular_noise
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity

    def get_next_pose_by_direction(self, pose, direction_angle, direction_length):
        turn = pose[2] - direction_angle
        turn = maths.convert_angle_to_minus_pi_pi(turn)
        rotation_duration = min(
            abs(turn) / self.max_angular_velocity, self.simulation_time_step
        )
        turn = np.sign(turn) * self.max_angular_velocity * rotation_duration
        turn += np.random.normal(0, self.angular_noise)
        move_duration = self.simulation_time_step - rotation_duration
        move = min(direction_length, self.max_linear_velocity * move_duration)

        return self.get_next_pose_by_turn_move(pose, turn, move)

    def get_next_pose_by_turn_move(self, pose, turn, move):
        new_pose = np.zeros(3)
        new_pose[2] = pose[2] - turn
        new_pose[0] = pose[0] + move * np.cos(new_pose[2])
        new_pose[1] = pose[1] + move * np.sin(new_pose[2])
        return new_pose

    def hard_constraint(self, pose, direction_angle, direction_length, other_poses):
        # new_position = pose[:2] + direction_length * np.array([np.cos(direction_angle), np.sin(direction_angle)])
        # distances_squared = np.sum((other_poses[:, :2] - new_position) ** 2, axis=1)
        distances_squared = np.sum((other_poses[:, :2] - pose[:2]) ** 2, axis=1)

        herd_hard_boundary = 1.0

        b_collision, d_theta, d_v = handle_collision_avoidance_hard(
            distances_squared, pose, other_poses, herd_hard_boundary
        )
        if b_collision:
            return self.get_next_pose_by_direction(pose, d_theta, d_v)

        herd_hard_boundary = 1.0

        new_pose = pose + direction_length * np.array(
            [np.cos(direction_angle), np.sin(direction_angle), 0.0]
        )
        distances_squared = np.sum((other_poses[:, :2] - new_pose[:2]) ** 2, axis=1)

        indexes = np.logical_and(
            0 < distances_squared,
            distances_squared <= (herd_hard_boundary) ** 2,
        )

        max_length = direction_length
        if np.any(indexes):

            for other_pose in other_poses[indexes]:
                b_int, int_position = maths.line_circle_intersection(
                    pose[:2],
                    new_pose[:2],
                    other_pose[:2],
                    herd_hard_boundary,
                )
                if b_int == 1:
                    max_length = min(
                        max_length, np.linalg.norm(pose[:2] - int_position)
                    )

        return self.get_next_pose_by_direction(pose, direction_angle, max_length)
