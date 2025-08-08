import numpy as np

def distance(a: list, b: list):
    distance = np.sqrt(np.sum((np.array(a[:2]) - np.array(b[:2])) ** 2))
    return distance

def clusterize(poses, threshold):
    coords=poses.tolist()
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