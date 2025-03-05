import copy


class Ant:
    def __init__(self, id):

        self.id = id
        self.tour = None
        self.cost = None

    def __repr__(self):
        return f"Ant ID: {self.id}, Tour: {self.tour}, Cost: {self.cost}"

    def get_tour(self):
        # Adds return path to tour
        tour = copy.copy(self.tour)
        tour.append(self.tour[0])
        return tour
