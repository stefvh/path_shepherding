import numpy as np
from shapely.geometry import MultiPoint, Point, LineString
from shapely.ops import unary_union

from utils.topology import ConcaveHull

def clusterize(robot_points, herd_points, threshold):
    n_robot_points = robot_points.shape[0]

    robots_index_above = np.array([None] * n_robot_points)
    robots_index_under = np.array([None] * n_robot_points)

    for i in range(n_robot_points):
        robot_point = robot_points[i]

        distances_to_herd = np.sqrt(np.sum((herd_points - robot_point) ** 2, axis=1))
        index_nearest = np.argmin(distances_to_herd)
        nearest_point = herd_points[index_nearest]

        p_tilde = robot_points - nearest_point
        heading = np.arctan2(
            robot_point[1] - nearest_point[1], robot_point[0] - nearest_point[0]
        )
        sign_heading = 1 if heading >= 0 else -1
        heading_others = heading - np.arctan2(p_tilde[:, 1], p_tilde[:, 0])
        indexes_above = np.logical_and(
            0 < sign_heading * heading_others,
            sign_heading * heading_others <= np.pi,
        )
        indexes_under = ~indexes_above
        indexes_under[i] = False
        indexes_above[i] = False

        distances_to_robots = np.sqrt(np.sum((robot_points - robot_point) ** 2, axis=1))
        # if i == 0:
        #     print(indexes_above)
        #     print(distances_to_robots)
        if np.any(indexes_above):
            index_above = np.argmin(distances_to_robots[indexes_above])
            index_above = np.argwhere(indexes_above)[index_above][0]
            # if i == 0:
            #     print(index_above[0])
            #     print(distances_to_robots[index_above])
            if distances_to_robots[index_above] < threshold:
                robots_index_above[i] = index_above
        if np.any(indexes_under):
            index_under = np.argmin(distances_to_robots[indexes_under])
            index_under = np.argwhere(indexes_under)[index_under][0]
            # if i == 0:
            #     print(index_under)
            #     print(distances_to_robots[index_under])
            if distances_to_robots[index_under] < threshold:
                robots_index_under[i] = index_under
    
    cluster_ids = np.array([1000] * n_robot_points)
    cluster_ids[0] = 0
    # Points that should not be in cluster -> id = -1
    # Points that are not investigated yet -> id = 1000

    focal_index = 17

    print(robots_index_above)
    print(robots_index_under)
    for i in range(n_robot_points):
        index_above = robots_index_above[i]
        index_under = robots_index_under[i]
        if index_above is None or index_under is None:
            pass # cluster_ids[i] = -1
        else:
            if i == focal_index:
                print("here 3")
                print("index_above: ", index_above)
                print("index_under: ", index_under)
                print("cluster_ids[i]: ", cluster_ids[i])
                print("cluster_ids[index_above]: ", cluster_ids[index_above])
                print("cluster_ids[index_under]: ", cluster_ids[index_under])
            if i == robots_index_under[index_above] and cluster_ids[index_above] >= 0:
                if i == focal_index:
                    print("here 4")
                if cluster_ids[i] > cluster_ids[index_above]:
                    if i == focal_index:
                        print("here 5")
                    np.where(cluster_ids == cluster_ids[i], cluster_ids[index_above], cluster_ids[i])
                elif cluster_ids[i] < cluster_ids[index_above]:
                    if i == focal_index:
                        print("here")
                        print("index_above: ", index_above)
                        print("cluster_ids[i]: ", cluster_ids[i])
                        print("cluster_ids[index_above]: ", cluster_ids[index_above])
                        print("cluster_ids: ", cluster_ids)
                    cluster_ids[index_above] = cluster_ids[i]
                    # np.where(cluster_ids == cluster_ids[index_above], cluster_ids[i])
                    if i == focal_index:
                        print("here 2")
                        print("cluster_ids: ", cluster_ids)
                else:
                    if i == focal_index:
                        print("here 6")
                    pass
            if i == robots_index_above[index_under] and cluster_ids[index_under] >= 0:
                if cluster_ids[i] > cluster_ids[index_under]:
                    np.where(cluster_ids == cluster_ids[i], cluster_ids[index_under], cluster_ids[i])
                elif cluster_ids[i] < cluster_ids[index_under]:
                    np.where(cluster_ids == cluster_ids[index_under], cluster_ids[i], cluster_ids[index_under])
                else:
                    pass

    # print(cluster_ids)

    clusters = []
    for cluster_id in np.unique(cluster_ids):
        if cluster_id >= 0:
            clusters.append(robot_points[cluster_ids == cluster_id])

    return clusters

def distance(a: list, b: list):
    distance = np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2))
    return distance

def clusterize_2(robot_points, herd_points, threshold):
    n_robot_points = robot_points.shape[0]

    robots_index_above = np.array([None] * n_robot_points)
    robots_index_under = np.array([None] * n_robot_points)

    unscanned_indices = list(range(n_robot_points))
    focus_indices = [0]

    cluster_ids = np.array([None] * n_robot_points)
    new_id = 0
    cluster_ids[0] = new_id

    while len(focus_indices):
        focus_index = focus_indices.pop()
        # print("----")
        # print("focus_index: ", focus_index)
        # print("unscanned_indices: ", unscanned_indices)
        unscanned_indices.remove(focus_index)

        focus_point = robot_points[focus_index]

        distances_to_herd = np.sqrt(np.sum((herd_points - focus_point) ** 2, axis=1))
        index_nearest = np.argmin(distances_to_herd)
        nearest_point = herd_points[index_nearest]

        p_tilde = robot_points - nearest_point
        heading = np.arctan2(
            focus_point[1] - nearest_point[1], focus_point[0] - nearest_point[0]
        )
        sign_heading = 1 if heading >= 0 else -1
        heading_others = heading - np.arctan2(p_tilde[:, 1], p_tilde[:, 0])
        indexes_above = np.logical_and(
            0 < sign_heading * heading_others,
            sign_heading * heading_others <= np.pi,
        )
        indexes_under = ~indexes_above
        indexes_under[focus_index] = False
        indexes_above[focus_index] = False

        distances_to_robots = np.sqrt(np.sum((robot_points - focus_point) ** 2, axis=1))
        if np.any(indexes_above):
            index_above = np.argmin(distances_to_robots[indexes_above])
            index_above = np.argwhere(indexes_above)[index_above][0]

            if distances_to_robots[index_above] <= threshold:
                robots_index_above[focus_index] = index_above

                if index_above in unscanned_indices:
                    if index_above not in focus_indices:
                        focus_indices.append(index_above)
                if cluster_ids[index_above] is None:
                    cluster_id = cluster_ids[focus_index]
                else:
                    cluster_id = min(cluster_ids[focus_index], cluster_ids[index_above])
                cluster_ids[index_above] = cluster_id
                cluster_ids[focus_index] = cluster_id
        
        if np.any(indexes_under):
            index_under = np.argmin(distances_to_robots[indexes_under])
            index_under = np.argwhere(indexes_under)[index_under][0]

            if distances_to_robots[index_under] <= threshold:
                robots_index_under[focus_index] = index_under

                if index_under in unscanned_indices:
                    if index_under not in focus_indices:
                        focus_indices.append(index_under)
                if cluster_ids[index_under] is None:
                    cluster_id = cluster_ids[focus_index]
                else:
                    cluster_id = min(cluster_ids[focus_index], cluster_ids[index_under])
                cluster_ids[index_under] = cluster_id
                cluster_ids[focus_index] = cluster_id

        if len(focus_indices) == 0 and len(unscanned_indices) > 0:
                new_id += 1
                focus_indices.append(unscanned_indices[0])
                cluster_ids[unscanned_indices[0]] = new_id
    
    # Recursively cut out points that do not have two neighbors
    to_scan_indices = []
    scanned_indices = []
    ria_toscan = [i for i in range(len(robots_index_above)) if robots_index_above[i] == None]
    riu_toscan = [i for i in range(len(robots_index_under)) if robots_index_under[i] == None]
    to_scan_indices.extend(ria_toscan)
    to_scan_indices.extend(riu_toscan)

    to_remove_indices = []

    while len(to_scan_indices) > 0:
        index = to_scan_indices.pop()
        scanned_indices.append(index)
        index_above = robots_index_above[index]
        index_under = robots_index_under[index]
        if (index_above is None) or (index_under is None) or (index_above in to_remove_indices) or (index_under in to_remove_indices):
            to_remove_indices.append(index)
            if index_above is not None:
                if index_above not in scanned_indices:
                    to_scan_indices.append(index_above)
            if index_under is not None:
                if index_under not in scanned_indices:
                    to_scan_indices.append(index_under)
    
    # Remove points that do not have two neighbors
    for index in to_remove_indices:
        cluster_ids[index] = -1

    clusters = []
    for cluster_id in np.unique(cluster_ids):
        if cluster_id >= 0:
            clusters.append(robot_points[cluster_ids == cluster_id])

    return clusters

def metric_caging_rate(threshold):
    def compute(t, poses_agent, poses_school):
        poses_agent_t = np.copy(poses_agent[t])
        poses_school_t = np.copy(poses_school[t])

        n_agent = poses_agent_t.shape[0]
        if n_agent < 3:
            return 0., clusters
        n_school = poses_school_t.shape[0]

        clusters = clusterize_2(poses_agent_t[:, :2], poses_school_t[:, :2], threshold)

        rates = [0.]
        polygons = []
        for cluster in clusters:
            if len(cluster) < 3:
                continue
            polygon = MultiPoint(cluster).convex_hull

            points = np.array(list(polygon.boundary.coords))
            shape_closed = True
            i = 0
            while i < len(points) and shape_closed:
                distance = np.linalg.norm(points[i] - points[(i + 1) % len(points)])
                shape_closed = distance <= threshold
                i += 1

            if not shape_closed:
                try:
                    concave_hull = ConcaveHull()
                    concave_hull.loadpoints(cluster)
                    concave_hull.calculatehull(tol=threshold)
                    polygon = concave_hull.boundary

                    points = np.array(list(polygon.boundary.coords))
                    shape_closed = True
                    i = 0
                    while i < len(points) and shape_closed:
                        distance = np.linalg.norm(points[i] - points[(i + 1) % len(points)])
                        shape_closed = distance <= threshold
                        i += 1
                except:
                    shape_closed = False

            polygons.append(polygon)

            if not shape_closed:
                rate = 0.
            else:
                n_enclosed = 0 

                indexes_to_remove = []
                for k, pose in enumerate(poses_school_t):
                    if polygon.contains(Point(pose[0], pose[1])):
                        n_enclosed += 1
                        indexes_to_remove.append(k)
                poses_school_t = np.delete(poses_school_t, indexes_to_remove, axis=0)
                rate = n_enclosed / n_school
            rates.append(rate)

        # TODO
        return np.sum(rates) #, clusters, polygons

    return compute

def cluster_points(points, threshold):
    coords=points.tolist()
    C=[]
    while len(coords):
        locus=coords.pop()

        begin_new_cluster = True
        continue_loop = True

        # See if locus can be added to an existing cluster
        i = 0
        while i < len(C) and continue_loop:
            cluster = C[i]
            j = 0
            while j < len(cluster) and continue_loop:
                x = cluster[j]
                if distance(locus, x) <= threshold:
                    cluster.append(locus)
                    begin_new_cluster = False
                    continue_loop = False
                j += 1
            i += 1
        
        # Otherwise create a new cluster
        if begin_new_cluster:
            cluster = [x for x in coords if distance(locus, x) <= threshold]
            C.append(cluster+[locus])
            for x in cluster:
                coords.remove(x)

    # Check if clusters should be merged
    i = 0
    while i < len(C):
        cluster = C[i]
        j = i + 1
        while j < len(C):
            other_cluster = C[j]
            if len(cluster) > 0 and len(other_cluster) > 0:
                k = 0
                searching = True
                while k < len(cluster) and searching:
                    l = 0
                    while l < len(other_cluster) and searching:
                        if distance(cluster[k], other_cluster[l]) <= threshold:
                            searching = False
                        l +=1
                    k +=1
                if not searching:
                    C[i] = cluster + other_cluster
                    C.pop(j)
                    j -= 1
            j += 1
        i += 1

    return C

def metric_caging_rate_old(threshold):
    def compute(t, poses_agent, poses_school):
        poses_agent_t = np.copy(poses_agent[t])
        poses_school_t = np.copy(poses_school[t])

        n_agent = poses_agent_t.shape[0]
        if n_agent < 3:
            return 0., clusters
        n_school = poses_school_t.shape[0]

        clusters = cluster_points(poses_agent_t[:, :2], threshold)

        rates = [0.]
        polygons = []
        for cluster in clusters:
            if len(cluster) < 3:
                continue
            polygon = MultiPoint(cluster).convex_hull

            points = np.array(list(polygon.boundary.coords))
            shape_closed = True
            i = 0
            while i < len(points) and shape_closed:
                distance = np.linalg.norm(points[i] - points[(i + 1) % len(points)])
                shape_closed = distance <= threshold
                i += 1

            if not shape_closed:
                try:
                    concave_hull = ConcaveHull()
                    concave_hull.loadpoints(cluster)
                    concave_hull.calculatehull(tol=threshold)
                    polygon = concave_hull.boundary

                    points = np.array(list(polygon.boundary.coords))
                    shape_closed = True
                    i = 0
                    while i < len(points) and shape_closed:
                        distance = np.linalg.norm(points[i] - points[(i + 1) % len(points)])
                        shape_closed = distance <= threshold
                        i += 1
                except:
                    shape_closed = False

            polygons.append(polygon)

            if not shape_closed:
                rate = 0.
            else:
                n_enclosed = 0 

                indexes_to_remove = []
                for k, pose in enumerate(poses_school_t):
                    if polygon.contains(Point(pose[0], pose[1])):
                        n_enclosed += 1
                        indexes_to_remove.append(k)
                poses_school_t = np.delete(poses_school_t, indexes_to_remove, axis=0)
                rate = n_enclosed / n_school
            rates.append(rate)

        # TODO
        return np.sum(rates) #, clusters, polygons

    return compute

# def metric_caging_rate(threshold):
#     def compute(t, poses_agent, poses_school):
#         poses_agent_t = np.copy(poses_agent[t])
#         poses_school_t = np.copy(poses_school[t])

#         n_agent = poses_agent_t.shape[0]
#         if n_agent < 3:
#             return 0.
#         n_school = poses_school_t.shape[0]

#         polygon = MultiPoint(poses_agent_t[:, :2]).convex_hull

#         points = np.array(list(polygon.boundary.coords))
#         shape_closed = True
#         i = 0
#         while i < len(points) and shape_closed:
#             distance = np.linalg.norm(points[i] - points[(i + 1) % len(points)])
#             shape_closed = distance <= threshold
#             i += 1

#         if not shape_closed:
#             try:
#                 concave_hull = ConcaveHull()
#                 concave_hull.loadpoints(poses_agent_t[:, :2])
#                 concave_hull.calculatehull(tol=threshold)
#                 polygon = concave_hull.boundary

#                 points = np.array(list(polygon.boundary.coords))
#                 shape_closed = True
#                 i = 0
#                 while i < len(points) and shape_closed:
#                     distance = np.linalg.norm(points[i] - points[(i + 1) % len(points)])
#                     shape_closed = distance <= threshold
#                     i += 1
#             except:
#                 shape_closed = False

#         if not shape_closed:
#             return 0.

#         n_enclosed = 0 
#         for pose in poses_school_t:
#             if polygon.contains(Point(pose[0], pose[1])):
#                 n_enclosed += 1
#         return n_enclosed / n_school

#     return compute


def metric_caging_probability(threshold):
    def compute(t, poses_agent, poses_school):
        poses_agent_t = np.copy(poses_agent[t])
        poses_school_t = np.copy(poses_school[t])

        n_agent = poses_agent_t.shape[0]
        if n_agent < 3:
            return 0.

        polygon = MultiPoint(poses_agent_t[:, :2]).convex_hull

        points = np.array(list(polygon.boundary.coords))
        shape_closed = True
        i = 0
        while i < len(points) and shape_closed:
            distance = np.linalg.norm(points[i] - points[(i + 1) % len(points)])
            shape_closed = distance <= threshold
            i += 1

        # try:
        #     polygon = MultiPoint(poses_agent_t[:, :2]).convex_hull

        #     points = np.array(list(polygon.boundary.coords))
        #     shape_closed = True
        #     i = 0
        #     while i < len(points) and shape_closed:
        #         distance = np.linalg.norm(points[i] - points[(i + 1) % len(points)])
        #         shape_closed = distance <= threshold
        #         i += 1
        # except:
        #     shape_closed = False

        if not shape_closed:
            try:
                concave_hull = ConcaveHull()
                concave_hull.loadpoints(poses_agent_t[:, :2])
                concave_hull.calculatehull(tol=threshold)
                polygon = concave_hull.boundary

                points = np.array(list(polygon.boundary.coords))
                shape_closed = True
                i = 0
                while i < len(points) and shape_closed:
                    distance = np.linalg.norm(points[i] - points[(i + 1) % len(points)])
                    shape_closed = distance <= threshold
                    i += 1
            except:
                shape_closed = False

        if not shape_closed:
            return 0.

        caging = True

        for pose in poses_school_t:
            caging = caging and polygon.contains(Point(pose[0], pose[1]))

        if caging:
            return 1.
        else:
            return 0.

    return compute


def metric_distance_upper_error(threshold):
    def compute(t, poses_agent, poses_school):
        poses_agent_t = np.copy(poses_agent[t])
        poses_school_t = np.copy(poses_school[t])

        error = 0.
        norm = 0.
        for pose in poses_agent_t:
            dist = np.sqrt(np.min(np.sum((poses_school_t[:, :2] - pose[:2]) ** 2, axis=1)))
            if dist - threshold > 0:
                error += dist - threshold
                norm += 1

        if norm > 0:
            return error / norm
        else:
            return 0.

    return compute


def metric_distance_lower_error(threshold):
    def compute(t, poses_agent, poses_school):
        poses_agent_t = np.copy(poses_agent[t])
        poses_school_t = np.copy(poses_school[t])
        n_school = poses_school_t.shape[0]

        error = 0.
        for pose in poses_agent_t:
            dist = np.sqrt(np.sum((poses_school_t[:, :2] - pose[:2]) ** 2, axis=1))
            error += np.sum(np.maximum(threshold - dist, 0))
        return error / n_school

    return compute


def metric_convex_enclosure_rate():
    def compute(t, poses_agent, poses_school):
        poses_agent_t = np.copy(poses_agent[t])
        poses_school_t = np.copy(poses_school[t])
        n_school = poses_school_t.shape[0]

        n_agent = poses_agent_t.shape[0]
        if n_agent < 3:
            return 0.

        polygon = MultiPoint(poses_agent_t[:, :2]).convex_hull

        n_enclosed = 0
        for pose in poses_school_t:
            if polygon.contains(Point(pose[0], pose[1])):
                n_enclosed += 1
        return n_enclosed / n_school

    return compute


def metric_max_nearest_agent_distance():
    def compute(t, poses_agent, poses_school):
        poses_agent_t = np.copy(poses_agent[t])

        max_dist = 0.
        for pose in poses_agent_t:
            dist = np.sqrt(np.sum((poses_agent_t[:, :2] - pose[:2]) ** 2, axis=1))
            max_dist = max(max_dist, dist.max())

        return max_dist

    return compute


def metric_min_caging_agents(threshold):
    def compute(t, _, poses_school):
        poses_school_t = np.copy(poses_school[t])

        herd_circles = []
        for school_pose in poses_school_t:
            herd_circles.append(Point(*school_pose[:2]).buffer(threshold))
        herd_union = unary_union(herd_circles)

        if herd_union.geom_type == 'MultiPolygon':
            return -1

        xs, ys = herd_union.exterior.xy

        N = 5
        samples = []
        for n in range(N):
            index = np.random.choice(len(xs), 1, replace=False)[0]

            start_x, start_y = xs[index], ys[index]

            finished = False
            points = [Point(start_x, start_y)]
            i = 0
            while not finished:
                a = points[-1]

                circle = LineString(np.column_stack(a.buffer(threshold).exterior.xy))
                intersections = circle.intersection(LineString(np.column_stack(herd_union.exterior.xy)))
                v, w = intersections[0], intersections[1]

                center = np.mean(poses_school_t[:, :2], axis=0)
                txy = np.array([a.x, a.y]) - center
                angle = np.arctan2(txy[1], txy[0])
                c, s = np.cos(angle), np.sin(angle)
                R = np.array(((c, -s), (s, c)))
                rel_v = (np.array([v.x, v.y]) - center).dot(R)

                if rel_v[1] < 0:
                    new_point = v
                else:
                    new_point = w

                points.append(new_point)

                i += 1
                if len(points) > 2 and np.sqrt(
                        (points[0].x - points[-1].x) ** 2 + (points[0].y - points[-1].y) ** 2) < threshold:
                    finished = True
                if i > 1000:
                    finished = True
            samples.append(len(points))

        return np.mean(samples)

    return compute
