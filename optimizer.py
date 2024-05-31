import sys
from copy import deepcopy
from bees import *



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
 
