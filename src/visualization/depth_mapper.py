import sys
from functools import partial
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# Import QApplication and the required widgets from PyQt5.QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow

from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QSlider
from qtrangeslider import QRangeSlider
from qtrangeslider import QLabeledRangeSlider
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QComboBox

from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QFormLayout
from PyQt5.QtWidgets import QGroupBox

import numpy as np

# Create a subclass of QMainWindow to setup the calculator's GUI
class DepthMapUi(QMainWindow):
    """PyCalc's View (GUI)."""

    def __init__(self):
        """View initializer."""
        super().__init__()
        # Set some main window's properties
        self.setWindowTitle('depth mapper')
        # self.setFixedSize(235, 235)
        # Set CSD canvas properties
        self.width = 5
        self.height = 20
        self.dpi = 100
        # separator info
        self.layerBorders = ['1/2', '2/3', '3/4', '4/5', '5/6']
        self.separatorBoolean = {border: False for border in self.layerBorders}
        self.separatorPosition = {border: 0 for border in self.layerBorders}
        # draw lines info
        self.lines = {border: None for border in self.layerBorders}
        self.labels = {border: None for border in self.layerBorders}
        # general widget info
        # Set the central widget and the general layout
        self.generalLayout = QGridLayout()
        self._centralWidget = QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.generalLayout)
        # Create the display and the buttons
        self._createCanvas()
        self._createSlider()
        self._createCheckList()
        self._createDropDown()
        self._createActionButtons()
        self._createDisplay()

    def _createCanvas(self):
        # creates MPL figure as QT compatible canvas
        self.fig = Figure(figsize=(self.width, self.height), dpi=self.dpi)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.generalLayout.addWidget(self.canvas, 1, 1, 4, 1)

    def _createSlider(self):
        self.slider = QSlider(Qt.Vertical)
        self.slider.setDisabled(True)
        self.slider.setTracking(False)
        self.generalLayout.addWidget(self.slider, 1, 2, 4, 1)

    def _selectSlider(self, index,):
        name = self.dropDown.itemText(index)
        position = self.separatorPosition[name]
        print(f'{index}: border {name} slider set at {position}')
        self.slider.setDisabled(False)
        self.slider.setValue(position)

    def _updateSlider(self, value):
        currentSlider = self.dropDown.currentText()
        self.separatorPosition[currentSlider] = value
        for sep, pos in self.separatorPosition.items():
            print(f'sep{sep} pos{pos}')
        print('\n')

    def _createCheckList(self):
        checkGroup = QGroupBox('Visible layer borders')
        checkLayout = QHBoxLayout()
        self.layerCheckBoxes = {f'{l}':'' for l in self.layerBorders}
        for layer in self.layerCheckBoxes.keys():
            checkbox = QCheckBox(layer)
            self.layerCheckBoxes[layer] = checkbox
            checkLayout.addWidget(checkbox)
        checkGroup.setLayout(checkLayout)
        self.generalLayout.addWidget(checkGroup, 1, 3, 1, 1)

    def _createDropDown(self):
        self.dropDown = QComboBox()
        self.generalLayout.addWidget(self.dropDown, 2, 3, 1, 1)

    def _updateDropDown(self, sepName, checkbox):
        print(f'updating dropdown with {sepName}')
        self.separatorBoolean[sepName] = checkbox.isChecked()
        self.dropDown.clear()
        for sepName, state in self.separatorBoolean.items():
            if state is True:
                self.dropDown.addItem(sepName)

        # set slider right acording to current dropdown state
        if not any(self.separatorBoolean.values()):
            self.slider.setValue(0)
            self.slider.setDisabled(True)
        else:
            self._selectSlider(self.dropDown.currentIndex())

        # erase lines and text if box unchecked
        for sepName, state in self.separatorBoolean.items():
            if state is False and self.lines[sepName] is not None:
                print(f'removing line for {sepName}')
                self.lines[sepName].remove()
                self.lines[sepName] = None
                self.labels[sepName].remove()
                self.labels[sepName] = None
                self.canvas.draw()

    def _createActionButtons(self):
        buttonGroup = QGroupBox('cellDB actions')
        buttonLayout = QHBoxLayout()

        self.saveButton = QPushButton('save layers')

        buttonLayout.addWidget(self.saveButton)
        buttonGroup.setLayout(buttonLayout)
        self.generalLayout.addWidget(buttonGroup, 3, 3, 1, 1)

    def _createDisplay(self):

        self.display = QLineEdit()
        self.display.setAlignment(Qt.AlignLeft)
        self.display.setReadOnly(True)
        self.generalLayout.addWidget(self.display, 4, 3, 1, 1)

    def _updateDisplay(self):
        return None

    def _drawSeparator(self, value):
        currentSlider = self.dropDown.currentText()

        # catches when the dropdown is empty
        if currentSlider == '':
            return

        # draw or redraw the line
        if self.lines[currentSlider] is not None:
            print('erasing old')
            self.lines[currentSlider].remove()
            self.labels[currentSlider].remove()

        print(f'drawing {currentSlider} at {value}')
        self.lines[currentSlider] = self.ax.axhline(value, color='white', linewidth=5)
        top, bottom = currentSlider.split('/')
        labelText = f'{top}\n{bottom}'
        self.labels[currentSlider] = self.ax.text(0, value-4, labelText, color='white', fontsize=40)

        self.canvas.draw()


class DepthMapCtrl():
    def __init__(self, model, view):
        self._view = view
        self._model = model
        self._model._drawCSD(self._view.ax)
        self._connectSignals()

    def testState(self, boxName):
        checkBox  = self._view.layerCheckBoxes[boxName]
        if checkBox.isChecked() == True:
            t = f'{boxName} on'
        else:
            t = f'{boxName} off'

        print(t)
        self._view.display.setText(t)
        return boxName

    def _connectSignals(self):
        # set the subset of separators to consider
        for boxName, checkBox in  self._view.layerCheckBoxes.items():
            checkBox.stateChanged.connect(partial(self._view._updateDropDown,
                                                  boxName, checkBox))

        self._view.dropDown.activated.connect(self._view._selectSlider)

        self._view.slider.valueChanged.connect(self._view._updateSlider)
        self._view.slider.valueChanged.connect(self._view._drawSeparator)


class DepthMapModel():
    def __init__(self):
        self.layerBorders = ['1/2', '2/3', '3/4', '4/5', '5/6']
        self.separatorBoolean = {border: False for border in self.layerBorders}
        self.separatorPosition = {border: 0 for border in self.layerBorders}
        self.lines = {}
        self.labels = {}
        self.file = ''
        self.csd = []
        self.currenteSeparator=''


    def _drawCSD(self, ax):
        # ax.imshow(np.random.rand(100, 25), aspect='equal', origin='lower', cmap='inferno')
        ax.imshow(np.zeros((100, 25)), aspect='equal', origin='lower', cmap='inferno')
        ax.get_xaxis().set_visible(False)
        ax.figure.tight_layout(pad=0)


    # def _drawSeparators(self, ax, separators):
    #     lines = list()
    #     labels = list()
    #     for separator,  position in separators.values():
    #         lines.append(ax.axhline(position))
    #         labels.append(ax.text(0, position, separator))
    #
    # def _clearSeparators(self, lines, lables):
    #     for line, label in zip(lines, lables):
    #         line.remove()
    #         label.remove()

# Client code
def main():
    """Main function."""
    # Create an instance of QApplication
    pycalc = QApplication(sys.argv)
    # Show the calculator's GUI
    view = DepthMapUi()
    view.show()
    # get model functions
    model = DepthMapModel()
    # create instance of the controller
    DepthMapCtrl(model=model, view=view)
    # Execute the calculator's main loop
    sys.exit(pycalc.exec_())

if __name__ == '__main__':
    main()





