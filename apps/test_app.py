from PyQt5 import Qt
import sys

class MainWindow(Qt.QMainWindow):

    def __init__(self, parent=None):
        Qt.QMainWindow.__init__(self, parent)

        layout = Qt.QVBoxLayout()
        self.combobox = Qt.QComboBox()
        self.combobox.setEnabled(0)
        button = Qt.QPushButton('init')
        layout.addWidget(button)
        layout.addWidget(self.combobox)

        self.frame = Qt.QFrame()
        self.frame.setLayout(layout)
        self.setCentralWidget(self.frame)

        # Signals and slots
        button.clicked.connect(self.on_button_click)
        self.combobox.currentTextChanged.connect(self.on_text_change)

        self.show()

    def on_button_click(self, s):
        print('button click')
        self.combobox.addItems(['a', 'b', 'c'])
        self.combobox.setEnabled(1)

    def on_text_change(self, s):
        print('text changed')
        print(s)

if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())