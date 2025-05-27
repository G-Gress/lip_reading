import sys
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout

app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle("PySide6 Test App")

layout = QVBoxLayout()
label = QLabel("Hello from PySide6!")
layout.addWidget(label)

window.setLayout(layout)
window.resize(300, 100)
window.show()

sys.exit(app.exec())
