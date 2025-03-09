import numpy as np
from Ant import Ant
import copy
import time


class ACO:
    def __init__(self, distance_matrix, max_iterations=100, n_ants=10, update_callback=None, evaporation_rate=0.5):

        self.distance_matrix = distance_matrix
        self.max_iterations = max_iterations
        self.n_ants = n_ants
        self.q = 10
        self.initial_pheromone = 10
        self.pheromone_weight = 1
        self.evaporation_rate = evaporation_rate

        rows, cols = self.distance_matrix.shape

        self.pheromone_matrix = self.initial_pheromone * np.ones((rows, cols))

        self.best_cost = []

        self.ants = [Ant(i) for i in range(n_ants)]
        self.best_solution = Ant(0)
        self.best_solution.cost = np.inf

        # This implementation is with help from chatGPT
        self.update_callback = update_callback

    # Algorithm inspired and mostly copied from matlab example from Vijander
    def run(self):
        # ACO Main loop
        start_time = time.time()
        for i in range(self.max_iterations):
            for k in range(self.n_ants):
                self.ants[k].tour = []

                for j in range(len(self.distance_matrix)):
                    probability = self.pheromone_matrix[:, j]**self.pheromone_weight

                    probability[self.ants[k].tour] = 0

                    probability = probability/np.sum(probability)

                    selected = self.roulette_wheel_selection(probability)

                    self.ants[k].tour.append(selected)

                self.calculate_cost(k)

                if self.ants[k].cost < self.best_solution.cost:
                    # Had to use deepcopy to copy the instance of the class and not reassigning it to a new memory space
                    self.best_solution = copy.deepcopy(self.ants[k])

            # Update pheromones
            self.update_pheromones()

            # Store best cost
            self.best_cost.append(self.best_solution.cost)

            print(f"Iteration = {i}, best cost = {self.best_solution.cost}")

        time_to_finish = time.time() - start_time

        if self.update_callback:
            self.update_callback(self.best_solution.get_tour(), self.best_solution.cost, self.pheromone_matrix, self.max_iterations, time_to_finish)



    def update_pheromones(self):
        self.evaporate()
        for i in range(self.n_ants):
            tour = self.ants[i].tour
            for j in range(len(tour) - 1):
                self.pheromone_matrix[tour[j], tour[j+1]] += self.q / self.ants[i].cost

    def evaporate(self):
        self.pheromone_matrix *= (1 - self.evaporation_rate)

    def roulette_wheel_selection(self, probability):
        """Ensure proper probability selection by avoiding numerical instability."""
        probability = np.nan_to_num(probability)
        probability /= np.sum(probability)

        r = np.random.rand()
        c = np.cumsum(probability)
        j = np.searchsorted(c, r)
        return j if j < len(probability) else len(probability) - 1

    def calculate_cost(self, iteration):
        n = len(self.ants[iteration].tour)

        tour = self.ants[iteration].get_tour()

        # Cost is total distance travelled
        self.ants[iteration].cost = 0
        for j in range(n):
            self.ants[iteration].cost += self.distance_matrix[tour[j], tour[j + 1]]
