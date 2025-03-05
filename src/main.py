import sys
from GUI import GUI
from PyQt6.QtWidgets import QApplication

# Load CSV file path
path = "/Users/hermangran/Documents/Programmering/Assignment-Part-C/Data/cities.csv"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = GUI(path)
    main_window.show()
    sys.exit(app.exec())