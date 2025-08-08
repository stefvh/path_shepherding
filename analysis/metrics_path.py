import numpy as np
from utils.clustering import clusterize

def _metric_individual_in_margin(poses, path):
    n_individual = poses.shape[0]

    x_min, x_max = path.x_range

    if not path.is_repositioned:
        return 1.
    
    sum = 0
    for pose in poses:
        if x_min < pose[0] < x_max:
            sum += path.is_in_margin(*pose[:2])
    return sum / n_individual

def _margin_error(path, x, y):
    is_y_above_bot = y > np.interp(x, path.bot_margin_x_space, path.bot_margin_y_space)
    is_y_below_top = y < np.interp(x, path.top_margin_x_space, path.top_margin_y_space)
    if is_y_above_bot and is_y_below_top:
        return 0
    else:
        if not is_y_above_bot:
            return np.interp(x, path.bot_margin_x_space, path.bot_margin_y_space) - y
        if not is_y_below_top:
            return y - np.interp(x, path.top_margin_x_space, path.top_margin_y_space)
        return 0

def _metric_individual_margin_error(poses, path):
    n_individual = poses.shape[0]

    x_min, x_max = path.x_range

    if not path.is_repositioned:
        return 0.
    
    sum = 0
    for pose in poses:
        if x_min < pose[0] < x_max:
            sum += _margin_error(path, *pose[:2])
            # sum += path.margin_error(*pose[:2])
    return sum / n_individual

def metric_school_in_margin():
    def compute(t, poses_agent, poses_school, path):
        poses_school_t = np.copy(poses_school[t])
        return _metric_individual_in_margin(poses_school_t, path)

    return compute

def metric_agents_in_margin():
    def compute(t, poses_agent, poses_school, path):
        poses_agent_t = np.copy(poses_agent[t])
        return _metric_individual_in_margin(poses_agent_t, path)

    return compute

def metric_school_out_margin(finish_radius=200.):
    def compute(t, poses_agent, poses_school, path):
        poses_school_t = np.copy(poses_school[t])

        total = poses_school_t.shape[0]

        if not path.is_repositioned:
            return total
        
        if path.is_at_finish(poses_school_t, finish_radius):
            return 0.
        
        sum = 0
        for pose in poses_school_t:
            sum += path.is_in_margin(*pose[:2])
        
        return total - sum

    return compute


def metric_school_out_margin_x_based(finish_radius=200.):
    def compute(poses_agent, poses_school, path):
        x_values = []
        y_values = []

        for t in range(poses_school.shape[0]):
            poses_school_t = np.copy(poses_school[t])

            center_x = np.mean(poses_school_t[:, 0])
            x_values.append(center_x)

            total = poses_school_t.shape[0]

            y_value = 0.
            if not path.is_repositioned:
                y_value = total
            else:
                if path.is_at_finish(poses_school_t, finish_radius):
                    y_value = 0.
                else:
                    sum = 0
                    for pose in poses_school_t:
                        sum += path.is_in_margin(*pose[:2])
                    y_value = total - sum
            y_values.append(y_value)
        return x_values, y_values

    return compute

def metric_school_out_margin_batched(time_step):
    def compute(t, poses_agent, poses_school, path):
        poses_school_t = np.copy(poses_school[t])
        total = poses_school_t.shape[0]

        if not path.is_repositioned:
            return total
        
        if path.is_at_finish(poses_school_t, 200.):
            return 0.

        t_max = poses_school.shape[0]
        if time_step < t < t_max - time_step:
            # mean_positions = np.mean(poses_school[t - time_step:t], axis=0)
            # sum = 0.
            # for pose in mean_positions:
            #     if not path.is_in_margin(*pose[:2]):
            #         sum += 1
            # return total - sum
            batch_positions = poses_school[t - time_step:t]
            batch_measurements = []
            for poses_school_t in batch_positions:
                sum = 0.
                for pose in poses_school_t:
                    sum += path.is_in_margin(*pose[:2])
                batch_measurements.append(total - sum)
            return np.mean(batch_measurements)
        return 0.
    
    return compute

def metric_school_margin_error():
    def compute(t, poses_agent, poses_school, path):
        poses_school_t = np.copy(poses_school[t])
        return _metric_individual_margin_error(poses_school_t, path)

    return compute

def metric_agents_margin_error():
    def compute(t, poses_agent, poses_school, path):
        poses_agent_t = np.copy(poses_agent[t])
        return _metric_individual_margin_error(poses_agent_t, path)

    return compute

def metric_time_school_finished_path():
    def compute(poses_agent, poses_school, path):
        _, x_max = path.x_range
        for t, poses_school_t in enumerate(poses_school):
            if np.all(poses_school_t[:, 0] > x_max):
                return t
        return -1

    return compute

def metric_time_agents_finished_path():
    def compute(poses_agent, poses_school, path):
        _, x_max = path.x_range
        for t, poses_agent_t in enumerate(poses_agent):
            if np.all(poses_agent_t[:, 0] > x_max):
                return t
        return -1

    return compute

def metric_school_in_direction_path():
    def compute(t, poses_agent, poses_school, path):
        n_school = poses_school.shape[1]
        sum = 0.
        if t > 0:
            for i in range(n_school):
                if poses_school[t, i, 0] - poses_school[t - 1, i, 0] >= 0:
                    sum += 1
        return sum / n_school

    return compute

def metric_school_in_direction_batched_path(time_step):
    def compute(t, poses_agent, poses_school, path):
        n_school = poses_school.shape[1]
        t_max = poses_school.shape[0]
        if time_step < t < t_max - time_step:
            previous_batch_position = np.mean(poses_school[t - time_step:t, :, 0])
            current_batch_position = np.mean(poses_school[t:t + time_step, :, 0])
            if current_batch_position - previous_batch_position >= 0:
                return 1.
            else:
                return 0.
        return 1.
    
    return compute

def metric_school_time_out_of_margin():
    x_buffer = 10.

    def compute(poses_agent, poses_school, path):
        t_max = poses_school.shape[0]
        sum = 0.
        for t, poses_school_t in enumerate(poses_school):
            if np.any(poses_school_t[:, 0] < path.x_range[0] + x_buffer) or np.any(poses_school_t[:, 0] > path.x_range[1] - x_buffer):
                pass
            else:
                if _metric_individual_in_margin(poses_school_t, path) == 1:
                    sum += 1
        return 1 - (sum / t_max)

    return compute

def metric_agents_time_out_of_margin():
    x_buffer = 10.

    def compute(poses_agent, poses_school, path):
        t_max = poses_agent.shape[0]
        sum = 0.
        for t, poses_agent_t in enumerate(poses_agent):
            if np.any(poses_agent_t[:, 0] < path.x_range[0] + x_buffer) or np.any(poses_agent_t[:, 0] > path.x_range[1] - x_buffer):
                pass
            else:
                if _metric_individual_in_margin(poses_agent_t, path) == 1:
                    sum += 1
        return 1 - (sum / t_max)

    return compute

def metric_herd_path_orientation(threshold):
    def compute(t, poses_agent, poses_school, path):
        poses_school_t = np.copy(poses_school[t])

        if not path.is_repositioned:
            return 0.
        
        if path.is_at_finish(poses_school_t, 200.):
            return 0.

        # clusterize school based on threshold
        clusters = clusterize(poses_school_t, threshold)

        # for every cluster; 
        #   compute the mean orientation
        #   get the path derivative at the school mean x-position
        errors = []
        for cluster in np.array(clusters):
            # Mean orientation
            sin_sum = np.sum(np.sin(cluster[:, 2]))
            cos_sum = np.sum(np.cos(cluster[:, 2]))
            mean_orientation = np.arctan2(sin_sum, cos_sum)
            # Approximate derivative
            mean_x = np.mean(cluster[:, 0])
            mean_y = np.mean(cluster[:, 1])
            distances = np.sqrt((mean_x - path.x_space) ** 2 + (mean_y - path.y_values) ** 2)
            index = np.argmin(distances)
            if index == path.x_space_num - 1:
                index = path.x_space_num - 2
            derivative = path.derivatives[index]
            # Compute error
            errors.append(mean_orientation - derivative)

        # average the errors between orientation and derivatve
        return np.mean(errors)
    
    return compute
