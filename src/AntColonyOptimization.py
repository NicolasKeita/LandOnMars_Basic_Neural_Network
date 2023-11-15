import numpy as np

# TODO move outside
distances = np.array([[np.inf, 2, 2, 5, 7],
                      [2, np.inf, 4, 8, 2],
                      [2, 4, np.inf, 1, 3],
                      [5, 8, 1, np.inf, 2],
                      [7, 2, 3, 2, np.inf]])

class AntColonyOptimization:
    def __init__(self, n_ants=3, evaporation_rate=0.95, alpha=1.0, beta=1.0):
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta

    def run(self, n_iterations):
        best_path = None
        all_time_best_path = ("placeholder", np.inf)
        for i in range(n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths, self.pheromone, self.all_inds, self.alpha, self.beta)
            self.pheromone * self.evaporation_rate
            self.ant_update_pheromone(all_paths, self.pheromone)
            self.pheromone * self.evaporation_rate
            if self.total_distance(all_time_best_path[0]) > self.total_distance(all_paths[0]):
                all_time_best_path = (all_paths[0], self.total_distance(all_paths[0]))
            best_path = all_time_best_path
        return best_path

    def gen_all_paths(self):
        pass

    def spread_pheromone(self, a, b, c, d, e):
        pass

    def ant_update_pheromone(self, a , b):

    def total_distance(self, idk):
        pass
