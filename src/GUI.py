import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QLineEdit, QHBoxLayout, QSlider, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import threading
import pandas as pd
from geopy.distance import geodesic
from ACO import ACO
from BPSO import BPSO


# Gui class displaying plots and adjusting algorithms more intuitively
class GUI(QMainWindow):

    def __init__(self, path, algorithm):
        super().__init__()
        # Setup Main Horizontal Layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Matplotlib Figure
        self.plot_layout = QVBoxLayout()
        self.main_layout.addLayout(self.plot_layout, stretch=3)

        # Stores algorithm
        self.algorithm = algorithm

        # For PSO
        if algorithm == "BPSO":
            self.setWindowTitle("Taxi allocation problem")

            self.df = pd.read_csv(path)
            self.num_cities = len(self.df)

            self.figure, self.ax = plt.subplots(figsize=(5, 8))
            self.canvas = FigureCanvas(self.figure)
            self.plot_layout.addWidget(self.canvas)

            self.customers = None

            self.plot_taxis()
            self.init_customers()

            # Settings BPSO Inputs & Labels
            self.settings_layout = QVBoxLayout()
            self.main_layout.addLayout(self.settings_layout, stretch=3)

            # Button to start ACO
            self.start_button = QPushButton("Find Best Allocation")
            self.start_button.clicked.connect(self.start_algorithm)
            self.settings_layout.addWidget(self.start_button)

            # Number of iterations
            layout_max_iterations, self.n_iterations = self.create_input_field("Iterations", 500, 1, 1000)
            self.settings_layout.addLayout(layout_max_iterations)

            # Number of Particles
            layout_particles, self.n_particles_input = self.create_input_field("Number of Particles", 10, 1, 100)
            self.settings_layout.addLayout(layout_particles)

            # Weight
            layout_weight, self.weight_factor = self.create_double_input("Weight Factor", 0.5, 0.0, 2)
            self.settings_layout.addLayout(layout_weight)

            # Cognitive factor
            layout_cognitive, self.cognitive_factor = self.create_double_input("Evaporation Rate", 2, 0, 10)
            self.settings_layout.addLayout(layout_cognitive)

            # Social factor
            layout_social, self.social_factor = self.create_double_input("Evaporation Rate", 2, 0, 10)
            self.settings_layout.addLayout(layout_social)

            # Displaying best cost
            self.best_cost_label = QLabel("Best Cost: N/A")
            self.settings_layout.addWidget(self.best_cost_label)

        # For ACO
        if algorithm == "ACO":
            self.setWindowTitle("Traveling Salesman Problem")

            self.df = pd.read_csv(path)
            self.num_cities = len(self.df)

            # Computes distance matrix
            self.cities = self.df[["Latitude", "Longitude"]].to_numpy()
            self.distance_matrix = np.zeros((self.num_cities, self.num_cities))

            for i in range(self.num_cities):
                for j in range(self.num_cities):
                    if i != j:
                        self.distance_matrix[i, j] = geodesic(self.cities[i], self.cities[j]).km

            self.figure, self.ax = plt.subplots(figsize=(5, 8))
            self.canvas = FigureCanvas(self.figure)
            self.plot_layout.addWidget(self.canvas)

            # Settings ACO Inputs & Labels
            self.settings_layout = QVBoxLayout()
            self.main_layout.addLayout(self.settings_layout, stretch=3)

            # Button to start ACO
            self.start_button = QPushButton("Find Best Route")
            self.start_button.clicked.connect(self.start_algorithm)
            self.settings_layout.addWidget(self.start_button)

            # Number of iterations
            layout_max_iterations, self.n_iterations = self.create_input_field("Iterations", 50, 1, 1000)
            self.settings_layout.addLayout(layout_max_iterations)

            # Number of Ants
            layout_ants, self.n_ants_input = self.create_input_field("Number of Ants", 10, 1, 100)
            self.settings_layout.addLayout(layout_ants)

            # Evaporation rate
            layout_evaporation, self.evaporation_rate = self.create_double_input("Evaporation Rate", 0.05, 0.0001, 1)
            self.settings_layout.addLayout(layout_evaporation)

            # Best Distance Label
            self.best_distance_label = QLabel("Best Distance: N/A")
            self.settings_layout.addWidget(self.best_distance_label)

            # Best Route Label
            self.best_route_label = QLabel("Best Route: N/A")
            self.best_route_label.setWordWrap(True)
            self.best_route_label.setFixedWidth(400)
            self.settings_layout.addWidget(self.best_route_label)

            # Pheromone matrix
            self.pheromone_matrix_label = QLabel("Pheromone Matrix")
            self.pheromone_matrix_label.setStyleSheet("font-size: 16px; font-weight: bold;")
            self.settings_layout.addWidget(self.pheromone_matrix_label)

            self.pheromone_matrix = QTableWidget(self.num_cities, self.num_cities)
            self.settings_layout.addWidget(self.pheromone_matrix)
            self.fill_table(self.pheromone_matrix, np.zeros((self.num_cities, self.num_cities)))

            # Store references to cost lines
            self.cost_lines = []

            # Initialize the plot
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
                line, = self.ax.plot(
                    [longitudes[i], longitudes[j]],
                    [latitudes[i], latitudes[j]],
                    'k-', linewidth=thickness, alpha=0.2, color='blue'
                )
                self.cost_lines.append(line)

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
        if self.algorithm == "ACO":
            threading.Thread(target=self.run_ACO, daemon=True).start()

        if self.algorithm == "BPSO":
            threading.Thread(target=self.run_BPSO(), daemon=True).start()

    def run_ACO(self):
        n_iterations = self.n_iterations.value()
        n_ants = self.n_ants_input.value()
        evaporation_rate = self.evaporation_rate.value()

        aco = ACO(self.distance_matrix, update_callback=self.update_GUI_ACO, max_iterations=n_iterations, n_ants=n_ants, evaporation_rate=evaporation_rate)
        aco.run()

    def run_BPSO(self):
        n_iterations = self.n_iterations.value()
        n_particles = self.n_particles_input.value()
        weight_factor = self.weight_factor.value()
        cognitive_factor = self.cognitive_factor.value()
        social_factor = self.social_factor

        # Starting algorithm
        bpso = BPSO(self.update_GUI_BPSO, self.df, self.customers, n_particles, n_iterations)
        bpso.run(weight_factor, cognitive_factor, social_factor)

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

    def update_GUI_ACO(self, best_tour, best_cost, matrix, iteration, time):

        for line in self.cost_lines:
            line.set_visible(False)

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

        # Update distance label
        self.best_distance_label.setText(f"Best Distance: {best_cost:.2f} km in {iteration + 1} iterations, time used: {time}s")
        self.best_route_label.setText(f"Best Route: {route_str}")

        # Update the pheromone matrix in the table
        self.fill_table(self.pheromone_matrix, matrix)

        # Redraw
        self.canvas.draw_idle()

    def update_GUI_BPSO(self, best_particle, iteration, time):
        # Remove old assignment lines if they exist
        if hasattr(self, "assignment_lines"):
            for line in self.assignment_lines:
                line.remove()
        self.assignment_lines = []

        # Draw new assignment lines for taxi-customer pairs (where position[i, j] == 1)
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if best_particle.position[i, j] == 1:
                    taxi_loc = self.df.loc[i, ["Longitude", "Latitude"]].to_numpy()
                    customer_loc = self.customers[j]
                    line, = self.ax.plot(
                        [taxi_loc[0], customer_loc[0]],
                        [taxi_loc[1], customer_loc[1]],
                        'k--', alpha=0.6
                    )
                    self.assignment_lines.append(line)

        self.best_cost_label.setText(f"Best Cost: {best_particle.cost:.2f} (Iteration {iteration}), time used: {time}s")
        self.canvas.draw_idle()

    def create_input_field(self, label_text, default_value, min_val, max_val):
        """Helper function to create labeled input fields."""
        layout = QHBoxLayout()

        label = QLabel(label_text)
        spin_box = QSpinBox()
        spin_box.setMinimum(min_val)
        spin_box.setMaximum(max_val)
        spin_box.setValue(default_value)

        layout.addWidget(label)
        layout.addWidget(spin_box)

        return layout, spin_box

    def create_double_input(self, label_text, default_value, min_val, max_val, step_size=0.0001):
        """Helper function to create labeled floating-point input fields."""
        layout = QHBoxLayout()

        label = QLabel(label_text)
        double_spin_box = QDoubleSpinBox()
        double_spin_box.setDecimals(4)  # ✅ Allows up to 4 decimal places
        double_spin_box.setSingleStep(step_size)  # ✅ Set precision step
        double_spin_box.setMinimum(min_val)
        double_spin_box.setMaximum(max_val)
        double_spin_box.setValue(default_value)

        layout.addWidget(label)
        layout.addWidget(double_spin_box)

        return layout, double_spin_box

    def plot_taxis(self):
        self.ax.clear()

        longitudes = self.df["Longitude"].to_numpy()
        latitudes = self.df["Latitude"].to_numpy()

        # Displays cities
        self.ax.scatter(longitudes, latitudes, c='red', marker='o')

        # Displays city names
        for i, name in enumerate(self.df["Name"]):
            self.ax.text(self.df["Longitude"][i], self.df["Latitude"][i], name, fontsize=9, ha='right', va='bottom')

        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        self.ax.set_title("Taxi")
        self.ax.grid(True)
        self.canvas.draw()

    def init_customers(self):

        # Upper and lower bounds
        max_longitude = max(self.df["Longitude"].to_numpy())
        min_longitude = min(self.df["Longitude"].to_numpy())
        max_latitude = max(self.df["Latitude"].to_numpy())
        min_latitude = min(self.df["Latitude"].to_numpy())

        self.customers = np.column_stack((
            np.random.uniform(min_longitude, max_longitude, self.num_cities),
            np.random.uniform(min_latitude, max_latitude, self.num_cities)
        ))
        self.ax.scatter(self.customers[:, 0], self.customers[:, 1])