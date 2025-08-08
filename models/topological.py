import numpy as np
from models.base import BaseModel
from utils import maths


class TopologicalModel(BaseModel):
    def __init__(self, 
                 **kwargs):
        super().__init__(**kwargs)
    
    def _get_herd_neighbors(self, i, poses_herd):
        pass
    
    def _get_robot_neighbors(self, i, poses_herd, poses_robots):
        pass