import numpy as np
from models.base import BaseModel
from utils import maths


class MetricModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_herd_neighbors(self, i, poses_herd):
        dist = np.sqrt(np.sum((poses_herd[:, :2] - poses_herd[i, :2]) ** 2, axis=1))
        return poses_herd[dist <= self.zor + self.zoo + self.zoa]
    
    def _get_robot_neighbors(self, i, poses_herd, poses_robots):
        dist = np.sqrt(np.sum((poses_robots[:, :2] - poses_herd[i, :2]) ** 2, axis=1))
        return poses_robots[dist <= self.zoi]
