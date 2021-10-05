import sys
import pathlib as pl
import warnings

cpp_path = pl.Path('/home/mateo/code/context_probe_analysis')
sys.path.append(str(cpp_path))
print(sys.path)

from src.metrics.dprime import ndarray_dprime

from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems_lbhb.baphy_io as io
from nems_lbhb.celldb import update_single_cell_data

import numpy as np
import json

from functools import partial
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
from PyQt5.QtWidgets import QRadioButton
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QComboBox

from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QFormLayout
from PyQt5.QtWidgets import QGroupBox
from PyQt5.QtWidgets import QButtonGroup


class OptoIdUi(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set some main window's properties
        self.setWindowTitle('depth mapper')
        # self.setFixedSize(235, 235)
        # Set canvas properties
        self.width = 15
        self.height = 30
        self.dpi = 100
        self.axes = np.empty([3], dtype='object')
        # general widget info
        # Set the central widget and the general layout
        self.generalLayout = QGridLayout()
        self._centralWidget = QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.generalLayout)
        # Create the display and the buttons
        self._createCanvas()
        self._createLoadBox()
        self._createDropDown()
        self._createNavigationButtons()
        self._createClasificationButtons()
        self._createExportButtons()
        # self._createDisplay()
        self.neuron_states = dict()

    def _createCanvas(self):
        # creates MPL figure as QT compatible canvas
        self.fig = Figure(figsize=(self.width, self.height), dpi=self.dpi)
        self.axes[0] = self.fig.add_subplot(211)
        self.axes[1] = self.fig.add_subplot(212, sharex=self.axes[0])
        self.axes[2] = inset_axes(self.axes[1], width="50%", height="50%", loc=1)
        # self.fig, self.axes = plt.subplots(2,1, squeeze=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.generalLayout.addWidget(self.canvas, 1, 1, 4, 1)

    def _createLoadBox(self):
        loadLayout = QHBoxLayout()
        self.loadButton = QPushButton('load')
        self.loadDisplay = QLineEdit()

        loadLayout.addWidget(self.loadButton)
        loadLayout.addWidget(self.loadDisplay)

        self.generalLayout.addLayout(loadLayout, 1, 2, 1, 1)

    def _createDropDown(self):
        self.dropDown = QComboBox()
        self.generalLayout.addWidget(self.dropDown, 2, 2, 1, 1)

    def _createNavigationButtons(self):

        self.prevButton = QPushButton('previous neuron')
        self.nextButton = QPushButton('next neuron')

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.prevButton)
        buttonLayout.addWidget(self.nextButton)

        self.generalLayout.addLayout(buttonLayout, 3, 2, 1, 1)

    def _createClasificationButtons(self):

        # todo make with a for loop and dicionaries.

        self.exiteButton = QPushButton('Excited')
        self.exiteButton.setCheckable(True)
        self.indifButton = QPushButton('Indifferent')
        self.indifButton.setCheckable(True)
        self.inhibButton = QPushButton('Inhibited')
        self.inhibButton.setCheckable(True)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.exiteButton)
        buttonLayout.addWidget(self.indifButton)
        buttonLayout.addWidget(self.inhibButton)

        self.classButtonGroup = QButtonGroup()
        self.classButtonGroup.setExclusive(True)
        self.classButtonGroup.addButton(self.exiteButton, 0)
        self.classButtonGroup.addButton(self.indifButton, 1)
        self.classButtonGroup.addButton(self.inhibButton, 2)

        self.state_map = {0: 'Excited', 1: 'Indifferent', 2: 'Inhibited'}

        self.generalLayout.addLayout(buttonLayout, 4, 2, 1, 1)

    def _createExportButtons(self):

        self.exportButton = QPushButton('export')

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.exportButton)

        self.generalLayout.addLayout(buttonLayout, 5, 2, 1, 1)

    def _populate_dropdown(self, neuron_list):
        self.dropDown.clear()
        self.neuron_states = dict()

        self.dropDown.addItems(neuron_list)
        self.neuron_states.fromkeys(neuron_list)

    def _update_gui_state(self, index):
        neuron = self.dropDown.itemText(index)

        if self.neuron_states[neuron] == 'Excited':
            self.exiteButton.setChecked(True)
        elif self.neuron_states[neuron] == 'Indifferent':
            self.indifButton.setChecked(True)
        elif self.neuron_states[neuron] == 'Inhibited':
            self.inhibButton.setChecked(True)
        elif self.neuron_states[neuron] == None:
            self.exiteButton.setChecked(False)
            self.indifButton.setChecked(False)
            self.inhibButton.setChecked(False)
        else:
            raise ValueError(f'unknown {self.neuron_states[neuron]} state')


class OptoIdModel():

    def __init__(self):
        self.rasterfs = 5000
        self.recache = False
        self.options = {'resp': True, 'rasterfs': self.rasterfs, 'stim':False}
        self.tstart = -0.02
        self.tend = 0.1
        # self.ready_recording()
        # self.sort_by_dprime()


    def ready_recording(self, parmfiles):
        self.parmfiles = parmfiles
        self.animal = self.parmfiles[0].split('/')[3]

        manager = BAPHYExperiment(parmfile=self.parmfiles)
        self.rec = manager.get_recording(recache=self.recache, **self.options)
        self.rec['resp'] = self.rec['resp'].rasterize()
        self.prestim = self.rec['resp'].extract_epoch('PreStimSilence').shape[-1] / self.rasterfs
        m = self.rec.copy().and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
        self.poststim = (self.rec['resp'].extract_epoch('REFERENCE', mask=m['mask'], allow_incomplete=True).shape[-1]
                         / self.rasterfs) + self.prestim
        self.lim = (self.tstart, self.tend)

        # get light on / off
        opt_data = self.rec['resp'].epoch_to_signal('LIGHTON')
        self.opto_mask = opt_data.extract_epoch('REFERENCE').any(axis=(1, 2))

        self.opt_s_stop = (np.argwhere(np.diff(opt_data.extract_epoch('REFERENCE')[self.opto_mask, :, :][0].squeeze())) + 1
                          ) / self.rasterfs

        self.sort_by_dprime()

    def sort_by_dprime(self):

        # Organizes data in an array with shape Optic_stim(off,on) x Trials x Neurons x Time

        # references = self.rec['resp'].extract_epoch('REFERENCE')
        # ref_light_on = references[self.opt_ref_mask,:,:]
        # ref_light_off = references[np.logical_not(self.opt_ref_mask),:,:]

        # same with just the light response part
        stim = self.rec['resp'].extract_epoch('Stim')
        stim_light_on = stim[self.opto_mask,:,:]
        stim_light_off = stim[np.logical_not(self.opto_mask),:,:]

        # determines difference between light on and light off trials and sorts neuron by max max effect
        opt_stim_dprime  = ndarray_dprime(stim_light_on, stim_light_off, axis=0)

        best_neuron_idx = np.flip(np.argsort(np.max(np.abs(opt_stim_dprime), axis=1)))

        self.neurons = np.asarray(self.rec['resp'].chans)
        self.sorted_neurons = np.asarray(self.neurons)[best_neuron_idx]


    def plot(self, neuron, axes=(None, )):

        raster = self.rec['resp'].extract_channels([neuron]).extract_epoch('REFERENCE').squeeze()
        mean_waveform = io.get_mean_spike_waveform(str(neuron), self.animal, usespkfile=True)

        if mean_waveform.size == 0:
            warnings.warn('waveform is an empty array')
            mean_waveform = np.zeros(2)

        if axes[0] == None:
            fig, axes = plt.subplots(2,1, squeeze=True, sharex=True)
            # add inset for mwf
            inset = inset_axes(axes[1], width="50%", height="50%", loc=1)
            axes = np.append(axes, inset)

        else:
            fig = axes[0].figure

        # r = rec['resp'].extract_channels([neuron]).extract_epoch('REFERENCE').squeeze()

        # psth
        on = raster[self.opto_mask, :].mean(axis=0) * self.options['rasterfs']
        on_sem = raster[self.opto_mask, :].std(axis=0) / np.sqrt(self.opto_mask.sum()) * self.options['rasterfs']
        t = np.arange(0, on.shape[-1] / self.options['rasterfs'], 1 / self.options['rasterfs']) - self.prestim
        axes[1].plot(t, on, color='blue')
        axes[1].fill_between(t, on - on_sem, on + on_sem, alpha=0.3, lw=0, color='blue')
        off = raster[~self.opto_mask, :].mean(axis=0) * self.options['rasterfs']
        off_sem = raster[~self.opto_mask, :].std(axis=0) / np.sqrt((~self.opto_mask).sum()) * self.options['rasterfs']
        t = np.arange(0, off.shape[-1] / self.options['rasterfs'], 1 / self.options['rasterfs']) - self.prestim
        axes[1].plot(t, off, color='grey')
        axes[1].fill_between(t, off - off_sem, off + off_sem, alpha=0.3, lw=0, color='grey')
        axes[1].set_ylabel('Spk / sec')
        # todo instead of setting limit, slice the data for proper  Y scaling.
        axes[1].set_xlim(self.lim[0], self.lim[1])

        # spike raster / light onset/offset
        st = np.where(raster[self.opto_mask, :])
        axes[0].scatter((st[1] / self.rasterfs) - self.prestim, st[0], s=1, color='b')
        offset = st[0].max()
        st = np.where(raster[~self.opto_mask, :])
        axes[0].scatter((st[1] / self.rasterfs) - self.prestim, st[0] + offset, s=1, color='grey')
        for ss in self.opt_s_stop:
            axes[0].axvline(ss - self.prestim, linestyle='--', color='lime')
        axes[0].set_title(neuron)
        axes[0].set_ylabel('Rep')
        axes[0].set_xlim(self.lim[0], self.lim[1])

        # plots waveform in inset
        axes[2].plot(mean_waveform, color='red')
        axes[2].axis('off')
        axes[2].set_xlabel('Time from light onset (sec)')

        return fig, axes


class OptoIdCtrl():
    def __init__(self, model, view):
        self._view = view
        self._model = model
        self._connectSignals()

    def ready_gui(self):
        print('readying UI')
        parmfiles = ['/auto/data/daq/Teonancatl/TNC014/TNC014a10_p_NON.m',
                     '/auto/data/daq/Teonancatl/TNC014/TNC014a11_p_NON.m']
        self._model.ready_recording(parmfiles)
        self._view._populate_dropdown(self._model.sorted_neurons)

    def update_gui_neuron(self, idx):
        print(idx)
        neuron = self._view.dropDown.currentText()
        print(f'{neuron} selected')
        self._model.plot(neuron, self._view.axes)

    def classify(self, id):
        neuron = self._view.dropDown.currentText()
        print(id)
        self._view.neuron_states[neuron] = self._view.state_map[id]

    def _connectSignals(self):
        # set the subset of separators to consider
        self._view.loadButton.clicked.connect(self.ready_gui)
        self._view.dropDown.activated.connect(self.update_gui_neuron)
        self._view.classButtonGroup.buttonClicked.connect(self.classify)
        self._view.classButtonGroup.buttonClicked.connect(lambda x:print(x))





# def main():
#     """Main function."""
#
#     # Create an instance of QApplication
#     optoId = QApplication(sys.argv)
#     # Show the calculator's GUI
#     view = OptoIdUi()
#     view.show()
#     # get model functions
#     model = OptoIdModel()
#     # create instance of the controller
#     OptoIdCtrl(model=model, view=view)
#     # Execute the calculator's main loop
#     sys.exit(optoId.exec_())


def main():

    parmfiles = ['/auto/data/daq/Teonancatl/TNC013/TNC013a10_p_NON.m',
                 '/auto/data/daq/Teonancatl/TNC013/TNC013a11_p_NON.m',
                 '/auto/data/daq/Teonancatl/TNC013/TNC013a12_p_NON.m']

    parmfiles = ['/auto/data/daq/Teonancatl/TNC014/TNC014a10_p_NON.m',
                 '/auto/data/daq/Teonancatl/TNC014/TNC014a11_p_NON.m']

    parmfiles = ['/auto/data/daq/Teonancatl/TNC015/TNC015a06_p_NON.m',
                 '/auto/data/daq/Teonancatl/TNC015/TNC015a12_p_NON.m']

    # # mean waveform not exported!
    # parmfiles = ['/auto/data/daq/Teonancatl/TNC016/TNC016a04_p_NON.m',
    #              '/auto/data/daq/Teonancatl/TNC016/TNC016a05_p_NON.m',
    #              '/auto/data/daq/Teonancatl/TNC016/TNC016a06_p_NON.m',
    #              '/auto/data/daq/Teonancatl/TNC016/TNC016a12_p_NON.m',
    #              ]

    parmfiles = ['/auto/data/daq/Teonancatl/TNC017/TNC017a12_p_NON.m']

    parmfiles = ['/auto/data/daq/Teonancatl/TNC018/TNC018a13_p_NON.m']

    ## FTC related NON files

    parmfiles = ['/auto/data/daq/Teonancatl/TNC014/TNC014a16_p_NON.m',
                 '/auto/data/daq/Teonancatl/TNC014/TNC014a17_p_NON.m']

    parmfiles = ['/auto/data/daq/Teonancatl/TNC015/TNC015a17_p_NON.m']

    parmfiles = ['/auto/data/daq/Teonancatl/TNC016/TNC016a17_p_NON.m']

    parmfiles = ['/auto/data/daq/Teonancatl/TNC017/TNC017a18_p_NON.m']

    parmfiles = ['/auto/data/daq/Teonancatl/TNC018/TNC018a20_p_NON.m']

    parmfiles = ['/auto/data/daq/Teonancatl/TNC019/TNC019a17_p_NON.m']


    hk_map = {'a': 'activated',
              'n': 'neutral',
              's': 'suppressed'}

    model = OptoIdModel()
    model.ready_recording(parmfiles)

    answers = dict()
    plt.ion()

    fig = plt.figure()
    axes = np.empty([3], dtype='object')
    axes[0] = fig.add_subplot(211)
    axes[1] = fig.add_subplot(212, sharex=axes[0])
    axes[2] = inset_axes(axes[1], width="50%", height="50%", loc=1)
    plt.draw()
    plt.pause(0.01)

    for neu in model.sorted_neurons[:]:
        for ax in axes:
            ax.clear()

        model.plot(neu, axes)

        plt.draw()
        plt.pause(0.01)
        # plt.show(block=False)
        while True:
            imp = input('[a]ctivated, [n]eutral, [s]uppressed: ')
            if imp in ['a', 'n', 's' , 'A', 'N', 'S']:
                imp = imp.lower()
                print(f'{neu} is {hk_map[imp]}')
                break
            else:
                print(f"Don't understan {imp}, use: a, n or s")

        update_single_cell_data(neu, phototag=imp)
        answers[neu] = imp

    # sorts dict back to cellid neuron order
    sort_ans = {key: answers[key] for key in model.neurons}

    linearized = json.dumps(sort_ans)
    print(linearized)

    return None

if __name__ == '__main__':
    main()
