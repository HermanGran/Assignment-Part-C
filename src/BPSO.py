import time

import numpy as np
from Particle import Particle
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import copy


class BPSO:

    def __init__(self, update_callback, df, customers, n_population, max_iterations):
        self.df = df
        self.n_taxis = len(self.df)
        self.n_customers = len(self.df)
        self.customers = []
        self.population = n_population
        self.max_iterations = max_iterations

        self.init_customers()
        self.taxis = self.df[["Longitude", "Latitude"]].to_numpy()

        self.distance_matrix = np.zeros((self.n_taxis, self.n_customers))
        for i in range(self.n_taxis):
            for j in range(self.n_customers):
                self.distance_matrix[i, j] = geodesic(self.taxis[i], self.customers[j]).km

        self.update_callback = update_callback

        self.particles = [Particle() for _ in range(self.population)]
        self.global_best = Particle()
        self.global_best.cost = np.inf
        self.init_particles()

    def init_customers(self):

        # Upper and lower bounds
        max_longitude = max(self.df["Longitude"].to_numpy())
        min_longitude = min(self.df["Longitude"].to_numpy())
        max_latitude = max(self.df["Latitude"].to_numpy())
        min_latitude = min(self.df["Latitude"].to_numpy())

        self.customers = np.column_stack((
            np.random.uniform(min_longitude, max_longitude, self.n_customers),
            np.random.uniform(min_latitude, max_latitude, self.n_customers)
        ))

    def init_particles(self):
        for particle in self.particles:

            # Init assignment
            particle.position = self.assignment()

            # Init velocity
            particle.velocity = np.zeros((self.n_taxis, self.n_customers))

            # Calculates Cost
            self.calculate_cost(particle)

            particle.best_position = particle.position
            particle.best_cost = particle.cost

            if particle.best_cost < self.global_best.cost:
                self.global_best = copy.deepcopy(particle)

    def assignment(self):
        assignment = np.zeros((self.n_taxis, self.n_customers))
        assigned_customer = np.random.permutation(self.n_customers)

        for taxi in range(self.n_taxis):
            assignment[taxi, assigned_customer[taxi]] = 1

        return assignment

    def calculate_cost(self, particle):
        particle.cost = np.sum(self.distance_matrix * particle.position)

    def calculate_velocity(self, particle):
        w = 0.5
        c1 = 2
        c2 = 2
        r1 = np.random.rand(self.n_taxis, self.n_customers)
        r2 = np.random.rand(self.n_taxis, self.n_customers)

        particle.velocity = (w * particle.velocity
                             + c1 * r1 * (particle.best_position - particle.position)
                             + c2 * r2 * (self.global_best.position - particle.position))

    def update_position(self, particle):
        P = 1 / (1 + np.exp(-particle.velocity))
        particle.position = np.zeros_like(particle.position)

        # List of available customer indices
        available_customers = list(range(self.n_customers))

        for i in range(self.n_taxis):
            if not available_customers:
                break  # Stop if no customers are left to assign
            # Get probabilities only for available customers
            row_probs = P[i, available_customers]
            # Choose the available customer with the highest probability
            max_index = np.argmax(row_probs)
            customer_index = available_customers.pop(max_index)
            particle.position[i, customer_index] = 1

    def run(self, w=0.5, c1=2, c2=2):
        start_time = time.time()
        # Main loop
        for i in range(self.max_iterations):
            for particle in self.particles:

                self.calculate_velocity(particle)

                self.update_position(particle)

                self.calculate_cost(particle)

                # Update Personal Best
                if particle.cost < particle.best_cost:
                    particle.best_cost = copy.deepcopy(particle.cost)
                    particle.best_position = copy.deepcopy(particle.position)

                    # Update Global Best
                    if particle.best_cost < self.global_best.cost:
                        self.global_best = copy.deepcopy(particle)

                print(f"Iteration: {i+1}, Particle cost: {particle.cost}, particle best cost: {particle.best_cost}, global best: {self.global_best.cost}")

        time_to_finish = time.time() - start_time

        if self.update_callback:
            self.update_callback(self.global_best, self.max_iterations, time_to_finish)
