import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import threading

import pandas as pd
from geopy.distance import geodesic
from ACO import ACO


class GUI(QMainWindow):

    def __init__(self, path):
        super().__init__()
        self.setWindowTitle("Traveling Salesman Problem")

        self.df = pd.read_csv(path) # Might move the path import
        self.num_cities = len(self.df)

        # Computes distance matrix
        self.cities = self.df[["Latitude", "Longitude"]].to_numpy()
        self.distance_matrix = np.zeros((self.num_cities, self.num_cities))

        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    # Calculate distance with the Haversine formula
                    self.distance_matrix[i, j] = geodesic(self.cities[i], self.cities[j]).km

        # Setup GUI layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Matplot figure
        self.figure, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Button to start ACO
        self.start_button = QPushButton("Find Best Route")
        self.start_button.clicked.connect(self.start_algorithm)
        self.layout.addWidget(self.start_button)

        # Label to display the best distance
        self.best_distance_label = QLabel("Best Distance: N/A")
        self.layout.addWidget(self.best_distance_label)

        # Distance matrix Table
        self.pheromone_matrix = QTableWidget(self.num_cities, self.num_cities)
        self.pheromone_matrix_label = QLabel("Pheromone Matrix")
        self.pheromone_matrix_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.layout.addWidget(self.pheromone_matrix_label)
        self.distance_matrix_info_label = QLabel("This matrix displays the pheromones between each city")
        self.layout.addWidget(self.distance_matrix_info_label)
        self.layout.addWidget(self.pheromone_matrix)
        self.fill_table(self.pheromone_matrix, np.zeros((self.num_cities, self.num_cities)))

        self.plot_cities()

    def plot_cities(self):
        self.ax.clear()

        longitudes = self.df["Longitude"].to_numpy()
        latitudes = self.df["Latitude"].to_numpy()

        max_distance = np.max(self.distance_matrix)

        # Displays the cost as thick lines
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                distance = self.distance_matrix[i, j]
                thickness = (distance / max_distance) * 5

                # Plots lines between each city
                self.ax.plot(
                    [longitudes[i], longitudes[j]],
                    [latitudes[i], latitudes[j]],
                    'k-', linewidth=thickness, alpha=0.2, color='blue'
                )

        # Displays cities
        self.ax.scatter(longitudes, latitudes, c='red', marker='o')

        # Displays city names
        for i, name in enumerate(self.df["Name"]):
            self.ax.text(self.df["Longitude"][i], self.df["Latitude"][i], name, fontsize=9, ha='right', va='bottom')

        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        self.ax.set_title("Cities in Norway")
        self.ax.grid(True)
        self.canvas.draw()

    def start_algorithm(self):
        # Got this tip from chatGPT: Run ACO in a separate thread to prevent GUI freezing
        threading.Thread(target=self.run_ACO, daemon=True).start()

    def run_ACO(self):
        aco = ACO(self.distance_matrix, update_callback=self.update_GUI)
        aco.run()

    def fill_table_new(self, table, matrix):
        """Update QTableWidget with city names and the pheromone matrix values."""
        table.setRowCount(self.num_cities)
        table.setColumnCount(self.num_cities)

        # Normalize pheromone values for better visualization
        max_pheromone = np.max(matrix)
        min_pheromone = np.min(matrix)
        range_pheromone = max_pheromone - min_pheromone
        if range_pheromone == 0:  # To avoid division by zero
            range_pheromone = 1

        # Set city names as headers
        table.setHorizontalHeaderLabels(self.df["Name"].tolist())
        table.setVerticalHeaderLabels(self.df["Name"].tolist())

        for i in range(self.num_cities):
            for j in range(self.num_cities):
                # Normalize pheromone values for display
                normalized_pheromone = (matrix[i][j] - min_pheromone) / range_pheromone
                item = QTableWidgetItem(f"{normalized_pheromone:.4f}")
                table.setItem(i, j, item)

    def fill_table(self, table,  matrix):
        table.setRowCount(self.num_cities)
        table.setColumnCount(self.num_cities)

        # Set city names as headers
        table.setHorizontalHeaderLabels(self.df["Name"].tolist())
        table.setVerticalHeaderLabels(self.df["Name"].tolist())

        for i in range(self.num_cities):
            for j in range(self.num_cities):
                item = QTableWidgetItem(f"{matrix[i][j]:.4f}")
                table.setItem(i, j, item)

    def update_GUI(self, best_tour, best_cost, matrix, iteration):
        """Updates the GUI dynamically while ACO runs."""

        # Update route dynamically
        if hasattr(self, "route_line"):
            self.route_line.set_data(
                [self.df["Longitude"][i] for i in best_tour],
                [self.df["Latitude"][i] for i in best_tour]
            )
        else:
            self.route_line, = self.ax.plot(
                [self.df["Longitude"][i] for i in best_tour],
                [self.df["Latitude"][i] for i in best_tour],
                'b-', linewidth=2, label="Best Route"
            )

        # Displaying route with city names
        tour = [self.df["Name"].iloc[i] for i in best_tour]

        # Print the route in a readable format
        route_str = " → ".join(tour)
        print(f"Best Route: {route_str}")

        # Update distance label
        self.best_distance_label.setText(f"Iteration {iteration + 1}: Best Distance = {best_cost:.2f} km \n"
                                         f"Best Route: \n"
                                         f"{route_str}")

        # Update the pheromone matrix in the table
        self.fill_table(self.pheromone_matrix, matrix)

        # Redraw
        self.canvas.draw_idle()
