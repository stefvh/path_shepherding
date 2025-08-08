import numpy as np
from utils import maths
from shapely.geometry import Point
from shapely.ops import nearest_points

# TODO: make fish more wild --> for density_maintaining --> see if cage helps

def get_default_model_params():
    return {
        "zone_width_repulsion": 1.0,
        "zone_width_alignment": 50.0,
        "zone_width_attraction": 50.0,
        "zone_width_aversion": 20.0,
        "zone_width_obstruction": 4.,
        "scaling_factor_repulsion": 100.0,
        "scaling_factor_alignment": 50.0,
        "scaling_factor_attraction": 1.0,
        "scaling_factor_aversion": 500.0,
        "scaling_factor_obstruction": 1000.0,
    }

class BaseModel:
    def __init__(self,
                 zone_width_repulsion,
                 zone_width_alignment,
                 zone_width_attraction,
                 zone_width_aversion,
                 zone_width_obstruction,
                 scaling_factor_repulsion,
                 scaling_factor_alignment,
                 scaling_factor_attraction,
                 scaling_factor_aversion,
                 scaling_factor_obstruction,):
        self.zor = zone_width_repulsion
        self.zoo = zone_width_alignment
        self.zoa = zone_width_attraction
        self.zoi = zone_width_aversion
        self.zob = zone_width_obstruction

        self.kor = scaling_factor_repulsion
        self.koo = scaling_factor_alignment
        self.koa = scaling_factor_attraction
        self.koi = scaling_factor_aversion
        self.kob = scaling_factor_obstruction

    def get_action(self, i, poses_herd, poses_robots, polygons_obstacles=None):
        force = self._get_action(i, poses_herd)
        if poses_robots.size > 0:
            force += self._get_action_aversive(i, poses_herd, poses_robots)
        if polygons_obstacles is not None:
            force += self._get_action_obstructed(i, poses_herd, polygons_obstacles)
        return force
    
    def _get_herd_neighbors(self, i, poses_herd):
        pass
    
    def _get_robot_neighbors(self, i, poses_herd, poses_robots):
        pass
    
    def _get_action(self, i, poses_herd):
        this_pose = poses_herd[i, :]
        poses_herd = self._get_herd_neighbors(i, poses_herd)
        dist = np.sqrt(np.sum((poses_herd[:, :2] - this_pose[:2]) ** 2, axis=1))

        j_r = np.logical_and(0.0 < dist, dist <= self.zor)
        j_o = np.logical_and(self.zor < dist, dist <= self.zor + self.zoo)
        j_a = np.logical_and(self.zor + self.zoo < dist, dist <= self.zor + self.zoo + self.zoa)

        f_r = np.zeros(2)
        if np.any(j_r):
            f_r = -np.sum(maths.normalize_matrix_rows(poses_herd[j_r, :2] - this_pose[:2]), axis=0)
        f_o = np.zeros(2)
        if np.any(j_o):
            f_o[0] = np.cos(this_pose[2]) + np.sum(np.cos(poses_herd[j_o, 2]), axis=0)
            f_o[1] = np.sin(this_pose[2]) + np.sum(np.sin(poses_herd[j_o, 2]), axis=0)
        f_a = np.zeros(2)
        if np.any(j_a):
            f_a = np.sum(maths.normalize_matrix_rows(poses_herd[j_a, :2] - this_pose[:2]), axis=0)
        
        f = self.kor * f_r + self.koo * f_o + self.koa * f_a

        return f

    def _get_action_aversive(self, i, poses_herd, poses_robots):
        poses_robots = self._get_robot_neighbors(i, poses_herd, poses_robots)
        this_pose = poses_herd[i, :]

        dist = np.sqrt(np.sum((poses_robots[:, :2] - this_pose[:2]) ** 2, axis=1))
        j_i = np.logical_and(0.0 < dist, dist <= self.zoi)

        f = 0
        if np.any(j_i):
            f += self.koi * -np.sum(maths.normalize_matrix_rows(poses_robots[j_i, :2] - this_pose[:2]), axis=0)
        
        return f
    
    def _get_action_obstructed(self, i, poses_herd, obstacle_dict):
        this_pose = poses_herd[i, :]

        polygons_wall = obstacle_dict["wall_polygons"]
        polygons_obstacles = obstacle_dict["polygons"]
        polygons = polygons_wall + polygons_obstacles

        f = 0
        for polygon in polygons:
            p = nearest_points(Point(this_pose[:2]), polygon)[1]
            q = np.array([p.x, p.y,]) - this_pose[:2]
            dist = np.sum(q ** 2, axis=0)
            if dist < self.zob ** 2:
                f += self.kob * -q / np.linalg.norm(q)

        return f
