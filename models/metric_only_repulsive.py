
from models.metric import MetricModel
import numpy as np

class MetricOnlyRepulsive(MetricModel):
    def get_action(self, i, poses_herd, poses_robots, polygons_obstacles=None):
        force = np.zeros(2)
        if poses_robots.size > 0:
            force += self._get_action_aversive(i, poses_herd, poses_robots)
        return force