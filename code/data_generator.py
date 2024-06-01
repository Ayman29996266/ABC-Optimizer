import numpy as np



def generate_dataset(min_val=-100, max_val=100, center=[0,0,0], dim=3, max_trails=500,
                     num_centroids=6, centroids_rad=30, centroids_min_dist=20,
                     av_cluster_size=36, av_cluster_rad=20, cluster_var_size=18,
                     cluster_var_rad=18, cluster_dots_min_dist=5):
    

    # ________________error_handelling:

    if min_val > max_val:
        raise ValueError ("min_val must be smaller than max_val")
    for i in center:
        if i < min_val:
            raise ValueError (f"data center element {i} is smaller than min_val")
        elif i > max_val:
            raise ValueError (f"data center element {i} is bigger than max_val")
    if centroids_min_dist > centroids_rad * 2:
        raise ValueError ("centroids_min_dist can't be bigger than the centroids_rad * 2")
    if av_cluster_rad > centroids_rad:
        raise ValueError ("av_cluster_rad can't be bigger than centroids_rad")
    if cluster_var_size > av_cluster_size:
        raise ValueError ("cluster_var_size can't be bigger than av_cluster_size")
    if cluster_var_rad > av_cluster_rad:
        raise ValueError ("cluster_var_rad can't be bigger than av_cluster_rad")
    if cluster_dots_min_dist > av_cluster_rad:
        raise ValueError ("cluster_dots_min_dist can't be bigger than av_cluster_rad")
    if len(center) != dim:
        raise ValueError ("length of center must equal the number of dimentions")
    
    
    # ________________data_ganeration:

    print("Procedural data generation...")

    center.insert(0, np.random.randint(0, 2))
    center.insert(0, 0)
    data = np.array([np.array(center), np.array(center)])

    for i in range(num_centroids):
        trails = 0
        centroid = np.array(1)
        for j in range(dim):
            centroid = np.append(centroid, center[j] + np.random.uniform(max(-centroids_rad, min_val),
                                                                         min(centroids_rad, max_val), 1))
        centroid = centroid[1:]
        while (np.any(np.linalg.norm(data[:, 2:] - centroid, axis=1) < centroids_min_dist)):
            centroid = np.array(1)
            for j in range(dim):
                centroid = np.append(centroid, center[j] + np.random.uniform(max(-centroids_rad, min_val),
                                                               min(centroids_rad, max_val), 1))
            centroid = centroid[1:]
            trails += 1
            if trails == max_trails:
                raise ValueError ("Can't generate data with current parameters (centroids).\nTry increasing centroids_rad or chose a center closer to the middle.\n Alternativaly, reduce centroids_min_dist or num_centroids.\nOr just try again...")

        centroid = np.insert(centroid, 0, np.random.randint(0, 2))
        centroid = np.insert(centroid, 0, i)

        cluster_size = av_cluster_size + np.random.randint(-cluster_var_size, cluster_var_size)
        cluster_radios = av_cluster_rad + np.random.randint(-cluster_var_rad, cluster_var_rad)
        dots = np.array([np.array(centroid), np.array(centroid)])
        for _ in range(cluster_size):
            trails = 0
            dot = np.array(1)
            for j in range(dim):
                dot = np.append(dot, centroid[j + 2] + np.random.uniform(max(-cluster_radios, min_val),
                                                                         min(cluster_radios, max_val), 1))
            dot = dot[1:]
            while (np.any(np.linalg.norm(dots[:, 2:] - dot, axis=1) < cluster_dots_min_dist)):
                dot = np.array(1)
                for j in range(dim):
                    dot = np.append(dot, centroid[j + 2] + np.random.uniform(max(-cluster_radios, min_val),
                                                                             min(cluster_radios, max_val), 1))
                dot = dot[1:]
                trails += 1
                if trails == max_trails:
                    raise ValueError ("Can't generate data with current parameters (dots).\nTry increasing av_cluster_rad.\n Alternativaly, reduce cluster_dots_min_dist or av_cluster_size\nOr just try again...")
                
            dot = np.insert(dot, 0, np.random.randint(0, 2))
            dot = np.insert(dot, 0, i)
            dots = np.vstack((dots, dot))
            
        data = np.vstack((data, dots[1:]))
    print("Done.\n")

    return np.array(data[2:])
