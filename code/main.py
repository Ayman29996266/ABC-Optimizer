from objective_function import SumOfSquaredErrors
from data_generator import generate_dataset
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from optimizer import *
import time



def color_list(num_colors):
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
              '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
              '#008080', '#e6beff', '#9a6324', '#800000', '#aaffc3',
              '#808000', '#ffd8b1', '#000075', '#808080', '#000000']
    while num_colors > len(colors):
        n_color = '#'
        for i in [np.random.choice(['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']) for _ in range(6)]:
            n_color += i
        colors.append(n_color)
    return colors[:num_colors]



def ABCO():
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Variables:

    print('\n-------------------------------------\n')
    number_of_iterations = 500
    optimizer = ABC()
    min_max_solution_pos = 100
    number_of_clusters = 12
    dataset = generate_dataset(min_val=-min_max_solution_pos, max_val=min_max_solution_pos, center=[0,0,0],
                               dim=3, max_trails=5000000, num_centroids=number_of_clusters, centroids_rad=100,
                               centroids_min_dist=20, av_cluster_size= 18, av_cluster_rad=32,
                               cluster_var_size=12, cluster_var_rad=12, cluster_dots_min_dist=1)

    colors = color_list(number_of_clusters)


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Clustering function optimization:

    start_time = time.time()

    sse = SumOfSquaredErrors(dim=number_of_clusters * 3,
                             n_clusters=number_of_clusters,
                             min=0, max=min_max_solution_pos)

    sse_optimization = optimizer.optimize(obj_function=sse,
                                          number_of_iterations=number_of_iterations,
                                          data=dataset[:, 2:], label='SSE clustering',
                                          colony_size=70, max_trials=8)

    sse_centroids = dict(enumerate(sse_optimization['optimal_solution'].pos.reshape(number_of_clusters, dataset[:, 2:].shape[1])))
    sse_clustering = []
    for instance in dataset[:, 2:]: 
        distances = []
        for idx in sse_centroids:
            distances.append(np.linalg.norm(instance - sse_centroids[idx]))
        sse_clustering.append([sorted(distances).index(element) for element in distances])
    
    print('Time: ', np.ceil((time.time() - start_time) * 1000), 'ms\n')


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Plotting:

    
    print('Preparing plots...')
    fig = plt.figure(figsize=(10, 10))

    gs = GridSpec(3, 3, width_ratios=[7, 1, 7],
                  height_ratios=[7, 1, 7])


    # ________________data_scatering:


    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax1.set_title('Data scatering')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=30, azim=10)

    for instance in dataset:
        ax1.scatter(instance[2], instance[3], instance[4],
                    s=30, edgecolor='w',
                    alpha=1, color='k') # to color according to cluster: colors[int(instance[0])]


    # ________________sse_clustering:


    ax2 = fig.add_subplot(gs[2], projection='3d')
    ax2.set_title('SSE clustering')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=30, azim=10)

    for instance, cluster in zip(dataset, sse_clustering):
        ax2.scatter(instance[2], instance[3], instance[4],
                    s=15, edgecolor='k',
                    alpha=1, color=colors[cluster[0]])
    


    # Uncoment the following to plot the centroids
    
    # for centroid in sse_centroids:
    #     ax2.scatter(sse_centroids[centroid][0],
    #                 sse_centroids[centroid][1],
    #                 sse_centroids[centroid][2],
    #                 color=colors[int(centroid)],
    #                 edgecolors='k', marker='o',lw=2, s=50)

    
    # ________________sse_fitness_track:


    ax3 = fig.add_subplot(gs[6])
    ax3.set_title('SSE fitness track')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Fitness')
    ax3.plot(range(number_of_iterations),
             sse_optimization['optimality_tracking'])


    plt.show()

    print('Done.\n')
    print('-------------------------------------')


if __name__ == "__main__":
    ABCO()

