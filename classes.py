import sys
from copy import deepcopy
from functions import *


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> OBJECTIVE FUNCTION:


class SumOfSquaredErrors():
    def __init__(self, dim, n_clusters, min=0.0, max=1.0):
        self.dim = dim
        self.min = min
        self.max = max
        self.n_clusters = n_clusters
        self.centroids = {}

    def sample(self):
        return np.random.uniform(low=self.min, high=self.max, size=self.dim)

    def evaluate(self, x, data):
        centroids = x.reshape(self.n_clusters, int(self.dim/self.n_clusters))
        self.centroids = dict(enumerate(centroids))
        centroid_distances = np.linalg.norm(data[:, None] - centroids, axis=2)
        clusters = {key: [] for key in self.centroids.keys()}
        clusters.fromkeys(clusters, [])
        for i in range(data.shape[0]):
            closest_centroid_idx = np.argmin(centroid_distances[i])
            clusters[closest_centroid_idx].append(data[i].tolist())

        sum_of_squared_errors = 0.0
        for idx in self.centroids:
            cluster_data = np.array(clusters[idx])
            if len(cluster_data) > 0:
                squared_distances = np.sum(np.power(cluster_data - self.centroids[idx], 2), axis=1)
                sum_of_squared_errors += np.sum(squared_distances)

        return sum_of_squared_errors


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ARTIFICIAL BEES:


class ArtificialBee():
    TRIAL_INITIAL_DEFAULT_VALUE = 0

    def __init__(self, obj_function, data=None):
        self.pos = obj_function.sample()
        self.obj_function = obj_function
        self.min = obj_function.min
        self.data = data
        self.max = obj_function.max
        self.fitness = obj_function.evaluate(self.pos, data)
        self.trial = ArtificialBee.TRIAL_INITIAL_DEFAULT_VALUE

    def update_bee(self, pos, fitness):
        if fitness <= self.fitness:
            self.pos = pos
            self.fitness = fitness
            self.trial = 0
        else:
            self.trial += 1

    def reset_bee(self, max_trials, optimal_solution):
        if self.trial >= max_trials and not self.pos is optimal_solution:
            self.pos = self.obj_function.sample()
            self.fitness = self.obj_function.evaluate(self.pos, self.data)
            self.trial = ArtificialBee.TRIAL_INITIAL_DEFAULT_VALUE


class EmployeeBee(ArtificialBee):
    def explore(self, max_trials):
        if self.trial <= max_trials:
            component = np.random.choice(self.pos)
            per = np.random.uniform(low=-1, high=1, size=len(self.pos))
            n_pos = self.pos + (self.pos - component) * per
            n_pos[n_pos < self.min] = self.min
            n_pos[n_pos > self.max] = self.max

            n_fitness = self.obj_function.evaluate(n_pos, self.data)
            self.update_bee(n_pos, n_fitness)

    def get_fitness(self):
        return self.fitness if self.fitness >= 0 else np.abs(self.fitness)

    def compute_prob(self, max_fitness):
        self.prob = self.get_fitness() / max_fitness


class OnLookerBee(ArtificialBee):
    def onlook(self, best_food_sources, max_trials):
        candidate = np.random.choice(best_food_sources)
        if self.trial <= max_trials:
            component = np.random.choice(candidate.pos)
            per = np.random.uniform(low=-1, high=1, size=len(candidate.pos))
            n_pos = candidate.pos + (candidate.pos - component) * per
            n_pos[n_pos < self.min] = self.min
            n_pos[n_pos > self.max] = self.max
                    
            n_fitness = self.obj_function.evaluate(n_pos, self.data)

            if n_fitness <= candidate.fitness:
                self.pos = n_pos
                self.fitness = n_fitness
                self.trial = 0
            else:
                self.trial += 1


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FINAL OPTIMIZATION CLASS:


class ABC():
    def __init__(self):
        self.no_of_optimizations = 0
        self.optimal_solution = None
        self.optimality_tracking = []
        self.solution_tracking = []
    
    def optimize(self, obj_function, colony_size=36,
                 number_of_iterations=36,
                 data=None, label=None, max_trials=12):
        print(f"{label} optimizing...")

        self.no_of_optimizations += 1
        if type(label) != str:
            label = self.no_of_optimizations

        # ________________reset_algorithm:
        self.optimal_solution = None
        self.optimality_tracking = []
        self.solution_tracking = []

        # ________________initialize_employees:
        self.employee_bees = []
        for _ in range(colony_size // 2):
            self.employee_bees.append(EmployeeBee(obj_function, data))

        # ________________initialize_onlookers
        self.onlokeer_bees = []
        for _ in range(colony_size // 2):
            self.onlokeer_bees.append(OnLookerBee(obj_function, data))


        for iteration in range(number_of_iterations):

            # ________________employee_bees_phase:
            for bee in self.employee_bees:
                bee.explore(max_trials)


            # ________________update_optimal_solution:
            n_optimal_solution = min(self.onlokeer_bees + self.employee_bees,
                                     key=lambda bee: bee.fitness)
            
            if not self.optimal_solution:
                self.optimal_solution = deepcopy(n_optimal_solution)
            else:
                if n_optimal_solution.fitness < self.optimal_solution.fitness:
                    self.optimal_solution = deepcopy(n_optimal_solution)


            # ________________calculate_probabilities:
            lis1 = []
            for bee in self.employee_bees:
                lis1.append(bee.get_fitness())
            sum_fitness = sum(lis1)
            for bee in self.employee_bees:
                bee.compute_prob(sum_fitness)

            # ________________select_best_food_sources:
            self.best_food_sources = list(filter(lambda bee: bee.prob > np.random.uniform(low=0, high=1), self.employee_bees))
            while not self.best_food_sources:
                self.best_food_sources = list(filter(lambda bee: bee.prob > np.random.uniform(low=0, high=1), self.employee_bees))

            # ________________onlooker_bees_phase:
            for bee in self.onlokeer_bees:
                bee.onlook(self.best_food_sources, max_trials)


            # ________________scout_bees_phase:
            for bee in self.onlokeer_bees + self.employee_bees:
                bee.reset_bee(max_trials, self.optimal_solution.pos)


            # ________________update_optimal_solution:
            n_optimal_solution = min(self.onlokeer_bees + self.employee_bees,
                                     key=lambda bee: bee.fitness)
            
            if not self.optimal_solution:
                self.optimal_solution = deepcopy(n_optimal_solution)
            elif n_optimal_solution.fitness < self.optimal_solution.fitness:
                self.optimal_solution = deepcopy(n_optimal_solution)


            # ________________update_optimality_tracking:
            self.optimality_tracking.append(self.optimal_solution.fitness)
            self.solution_tracking.append(self.optimal_solution.pos)


            # ________________progress_message:

            sys.stdout.write('%s\r' % f"{label} progress: {np.ceil((iteration / number_of_iterations) * 100)}%.")
            sys.stdout.flush()
        
        print(f"{label} done.            \nFirst sulotion distance: {self.optimality_tracking[0]}")
        print(f"Final sulotion distance: {self.optimality_tracking[-1]}\n")
        return {'optimal_solution': self.optimal_solution, 'optimality_tracking': self.optimality_tracking, 'solution_tracking': self.solution_tracking}


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SIMULATOR:


class Simulator():
    def __init__(self, data, sse=None, budget=36, demo=10):
        print('Simulating...')
        self.demo = demo  
        self.no_of_clusters = len(sse[0])
        self.budget = budget

        sse_data = [list(element1) + list(element2) for element1, element2 in zip(sse, data[:, 1:])]
        self.sse_clusters = []
        for ele in sse_data:
            while len(self.sse_clusters) <= int(ele[0]) + 1:
                self.sse_clusters.append([])
            self.sse_clusters[int(ele[0])].append(ele)
        self.sse = self.__clus_sim(self.sse_clusters)
        print("SSE done.")

        self.non = self.__non_targeted()
        print("No targeting done.")
        
        self.demo = self.__demographic()
        print("Demographic done.")        

        print('Simulation completed.\n')

    def __clus_sim(self, clusters):
        percents = []
        for cluster in clusters:
            adds = []
            subs = []
            for ele in cluster:
                if ele[self.no_of_clusters] != 0:
                    subs.append(ele[self.no_of_clusters + 1:])
                else:
                    adds.append(ele)

            for add in adds:
                clicks = 0
                dists = []
                add = np.array(add)

                if self.budget > len(subs):
                    while self.budget > len(subs):
                        for ele in cluster:
                            if ele[self.no_of_clusters] < 2:
                                subs.append(ele[self.no_of_clusters + 1:])

                chosens = np.random.choice(range(len(subs)),
                                           size=self.budget,
                                           replace=False)
                
                add = add[self.no_of_clusters + 1:]
                for sub in [subs[i] for i in chosens]:
                    sub = np.array(sub)
                    dists.append(np.linalg.norm(add - sub))

                dists = np.array([dists]).T
                probs = 1 - (dists / np.linalg.norm(np.array([100, 100, 100]) - np.array([0, 0, 0])))

                for i in probs:
                    if np.random.rand() <= i:
                        clicks += 1

                percents.append((clicks / self.budget) * 100)

        return sum(percents) / len(percents)

    def __non_targeted(self):
        percents = []
        adds = []
        subs = []
        for cluster in self.sse_clusters:
            for ele in cluster:
                if ele[self.no_of_clusters] != 0:
                    subs.append(ele[self.no_of_clusters + 1:])
                else:
                    adds.append(ele[self.no_of_clusters + 1:])

        for add in adds:
            clicks = 0
            dists = []
            add = np.array(add)
            chosens = np.random.choice(range(len(subs)),
                                       size=self.budget,
                                       replace=True)
            
            for sub in [subs[i] for i in chosens]:
                sub = np.array(sub)
                dists.append(np.linalg.norm(add - sub))

            dists = np.array([dists]).T
            probs = 1 - (dists / np.linalg.norm(np.array([100, 100, 100]) - np.array([0, 0, 0])))

            for i in probs:
                if np.random.rand() <= i:
                    clicks += 1

            percents.append((clicks / self.budget) * 100)

        return sum(percents) / len(percents)

    def __demographic(self):
        percents = []
        adds = []
        subs = []
        for cluster in self.sse_clusters:
            for ele in cluster:
                if ele[self.no_of_clusters] != 0:
                    subs.append(ele[self.no_of_clusters + 1:])
                else:
                    adds.append(ele[self.no_of_clusters + 1:])

        adds = np.array(adds)
        subs = np.array(subs)
        for add in adds:
            clicks = 0
            dists = []
            add = np.array(add)
            chosens = np.where((subs[:, 0] < add[0] + self.demo) & (subs[:, 0] > add[0] - self.demo))[0]
            chosens = np.concatenate((np.where((subs[:, 1] < add[1] + self.demo) & (subs[:, 1] > add[1] - self.demo))[0], chosens))
            chosens = np.concatenate((np.where((subs[:, 2] < add[2] + self.demo) & (subs[:, 2] > add[2] - self.demo))[0], chosens))


            while chosens.size < self.budget:
                self.demo += 1
                chosens = np.where((subs[:, 0] < add[0] + self.demo) & (subs[:, 0] > add[0] - self.demo))[0]
                chosens = np.concatenate((np.where((subs[:, 1] < add[1] + self.demo) & (subs[:, 1] > add[1] - self.demo))[0], chosens))
                chosens = np.concatenate((np.where((subs[:, 2] < add[2] + self.demo) & (subs[:, 2] > add[2] - self.demo))[0], chosens))
                if self.demo >= self.budget:
                    break

            
            while chosens.size > self.budget:
                chosens = chosens[:-1]


            for sub in [subs[i] for i in chosens]:
                sub = np.array(sub)
                dists.append(np.linalg.norm(add - sub))

            dists = np.array([dists]).T
            probs = 1 - (dists / np.linalg.norm(np.array([100, 100, 100]) - np.array([0, 0, 0])))

            for i in probs:
                if np.random.rand() <= i:
                    clicks += 1

            percents.append((clicks / self.budget) * 100)

        return sum(percents) / len(percents)
