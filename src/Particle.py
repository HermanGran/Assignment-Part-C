import numpy as np


class Particle:

    def __init__(self):
        self.position = None
        self.velocity = None
        self.cost = None
        self.best_position = None
        self.best_cost = None

    def __repr__(self):
        return f""" Particle
        Position: {self.position}
        Velocity: {self.velocity}
        Cost: {self.cost}
        Best pos: {self.best_position}
        Best cost: {self.cost}
        """
