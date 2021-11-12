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

        self.layerBorders = ['1/2', '2/3', '3/4', '4/5', '5/6']

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




class DepthMapModel():
    def __init__(self):
        # separator info
        self.landmarks = ['1/2', '2/3', '3/4', '4/5', '5/6']
        self.landmarkPosition = {border: 0 for border in self.landmarks}
        self.landmarkBoolean = {border: False for border in self.landmarks}
        self.lines = list()
        self.file = ''
        self.csd = []

    def _drawCSD(self, ax):
        # ax.imshow(np.random.rand(100, 25), aspect='equal', origin='lower', cmap='inferno')
        ax.imshow(np.zeros((100, 25)), aspect='equal', origin='lower', cmap='inferno')
        ax.get_xaxis().set_visible(False)
        ax.figure.tight_layout(pad=0)

    def errase_lines(self):
        print('errasing lines...')
        while len(self.lines) != 0:
            art = self.lines.pop()
            art.remove()
        print('done')

    def draw_lines(self, ax):
        self.errase_lines()
        print('drawing lines...')
        for sName in self.landmarks:
            sBool = self.landmarkBoolean[sName]
            sPos = self.landmarkPosition[sName]
            if sBool:
                self.lines.append(ax.axhline(sPos, color='white', linewidth=5))
                top, bottom = sName.split('/')
                labelText = f'{top}\n{bottom}'
                text_ypos = sPos - 4 # ToDo find smarter way of labeling lines
                self.lines.append(ax.text(0, text_ypos, labelText, color='white', fontsize=40))
        print('done')


class DepthMapCtrl():
    def __init__(self, model, view):
        self._view = view
        self._model = model
        self._model._drawCSD(self._view.ax)
        self._connectSignals()

    def _updateDropDown(self, sepName, checkbox):
        print(f'updating dropdown with {sepName}...')
        self._model.landmarkBoolean[sepName] = checkbox.isChecked()
        self._view.dropDown.clear()
        for sepName, sBool in self._model.landmarkBoolean.items():
            if sBool:
                self._view.dropDown.addItem(sepName)

        # set slider right according to current dropdown state
        if not any(self._model.landmarkBoolean.values()):
            self._view.slider.setValue(0)
            self._view.slider.setDisabled(True)
        else:
            self._selectSlider(self._view.dropDown.currentIndex())

        # erase lines and text if box unchecked
        self._model.draw_lines(self._view.ax)
        self._view.canvas.draw()
        print('done')

    def _selectSlider(self, index,):
        name = self._view.dropDown.itemText(index)
        position = self._model.landmarkPosition[name]
        print(f'{index}: border {name} slider set at {position}')
        self._view.slider.setDisabled(False)
        self._view.slider.setValue(position)

    def _updateSlider(self, value):
        currentSlider = self._view.dropDown.currentText()
        self._model.landmarkPosition[currentSlider] = value
        for sep, pos in self._model.landmarkPosition.items():
            print(f'sep:{sep}, pos:{pos}')

        self._model.draw_lines(self._view.ax)
        self._view.canvas.draw()

    def _connectSignals(self):
        # set the subset of separators to consider
        for boxName, checkBox in  self._view.layerCheckBoxes.items():
            checkBox.stateChanged.connect(partial(self._updateDropDown,
                                                  boxName, checkBox))

        self._view.dropDown.activated.connect(self._selectSlider)
        self._view.slider.valueChanged.connect(self._updateSlider)


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
    ctrl = DepthMapCtrl(model=model, view=view)
    # Execute the calculator's main loop
    sys.exit(pycalc.exec_())

if __name__ == '__main__':
    main()





