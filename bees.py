import numpy as np



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
 