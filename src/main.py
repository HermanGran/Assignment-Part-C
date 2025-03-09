import sys
from GUI import GUI
from PyQt6.QtWidgets import QApplication


# Load CSV file path
city = "/Users/hermangran/Documents/Programmering/Assignment-Part-C/Data/cities.csv"
taxi = "/Users/hermangran/Documents/Programmering/Assignment-Part-C/Data/taxi.csv"

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Run below for BPSO Algorithm, comment out the other
    main_window = GUI(taxi,"BPSO")

    # Run below for ACO Algorithm, comment out the other
    # main_window = GUI(city, "ACO")
    main_window.show()
    sys.exit(app.exec())
