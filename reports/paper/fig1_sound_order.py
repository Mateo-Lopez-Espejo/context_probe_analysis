import numpy as np
import pyaudio
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal

import plotly.graph_objects as go
from plotly.subplots import make_subplots
plt.style.use('bmh')

from src.root_path import root_path

"""
This is a superfluous tool to embed recorded sounds in the main paper of my PhD.
On runtime it launches a monitor that displays the current sound input of the default mic of the system
On pressing enter it saves a 1s waveform inmediatly after the key press
after 4 Sounds have been stored it asks to close the monitor window to continue, or to press Enter again to overwrite
the firts wave recorded (working as a blackbox) 
"""

# output folder, change me!
folder = root_path / 'reports' / 'figures' / 'paper'
folder.mkdir(parents=True, exist_ok=True)


SAMPLESIZE = 4096 # number of data points to read at a time
SAMPLERATE = 44100 # time resolution of the recording device (Hz)
# SAMPLERATE = 11000 # time resolution of the recording device (Hz)

p = pyaudio.PyAudio() # instantiate PyAudio
stream=p.open(format=pyaudio.paInt16,channels=1,rate=SAMPLERATE,input=True,
              frames_per_buffer=SAMPLESIZE) # use default input device to open audio stream

# set up plotting
fig = plt.figure()
ax = plt.axes(xlim=(0, SAMPLESIZE-1), ylim=(-9999, 9999))
line, = ax.plot([], [], lw=1)

# x axis data points
x = np.linspace(0, SAMPLESIZE-1, SAMPLESIZE)

waves = list()

# function to capture a waveform on Enter press
def onkey(event):
    waves.append(np.frombuffer(stream.read(SAMPLERATE), dtype=np.int16))
    if len(waves) > 4:
        _ = waves.pop(0)
    print(f'stored sound {len(waves)}')

# methods for animation
def init():
    line.set_data([], [])
    return line,

def animate(i):
    y = np.frombuffer(stream.read(SAMPLESIZE), dtype=np.int16)
    line.set_data(x, y)

    if len(waves) >= 4:
        print('saved 4 figures, close the plot to continue or overwrite the first figure')\

    return line,

anim = FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)
cid = fig.canvas.mpl_connect('key_press_event', onkey)

plt.show()
# stop and close the audio stream
stream.stop_stream()
stream.close()
p.terminate()

print(f'captured waves with shapes {[ww.shape for ww in waves]} ')


## now that we have raw waveforms, preprocess them a bit, the waveforms, embed them in publication quality figures
def preprocess(wave):
    # lowpass, decimates, and normalizes
    # sos = signal.butter(4, 5000, 'low', fs=SAMPLERATE, output='sos')
    # filtered = signal.sosfilt(sos, wave)
    # roughly decimate to the denominator samplig rate
    decimated = signal.decimate(wave,int(SAMPLERATE/2000))
    decimated = decimated / np.max(np.absolute(decimated))
    return decimated

# # lowpassfilter and downsample waveforms
waves = [preprocess(ww) for ww in waves]

fig, ax = plt.subplots()
ax.plot(np.stack(waves, axis=1)+np.arange(4))
print('these are the waves as they will be displayed in the refined figure\n'
      'close the figure to continue, or restart the script to get new waves')
plt.show()

# adds silence
waves.insert(0,np.zeros(waves[0].shape))
n_samps = waves[0].shape[0]


# hard coded perfect cover order
sequences = np.asarray([[0,1,3,2,4,4],
                        [0,3,4,1,1,2],
                        [0,4,2,3,3,1],
                        [0,2,2,1,4,3]])
eg_probe = 4

# colors = FOURCOLOR

Red = '#E15759'
Teal = '#76B7B2'
Yellow = '#EDC948'
Brown = '#9C755F'
Grey = '#BAB0AC'
colors = [Grey, Yellow, Red, Teal, Brown]

xbox = np.asarray([0,1,1,0,0])
color_box_height = 0.75
ybox = (np.asarray([0,0,1,1,0])-0.5) * color_box_height

# ensures normalized waves fit snug in boxes
waves = [ww*0.5*color_box_height for ww in waves]


egbox_height = 0.8

all_figs = list()

fig = go.Figure()
all_figs.append(fig)
for ss, seq in enumerate(sequences):
    for ww, wave_idx in enumerate(seq):
        color = colors[wave_idx]
        if ww > 0:
            # Colored boxes except silence
            _ = fig.add_trace(go.Scatter(x=xbox+ww, y=ybox+ss, fill='toself',
                                         mode='lines',
                                         line=dict(width=1,
                                                   color='gray'),
                                         fillcolor=color,
                                         showlegend=False)
                              )

        # wave form plots
        x = np.linspace(0,1,n_samps) + ww
        y = waves[wave_idx] + ss
        _ = fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                     line=dict(color='black',
                                               width=1),
                                     showlegend=False,
                                     )
                          )

# Add e.g. dotted boxes, ensure are the last to be drawn so they are on top
for ss, seq in enumerate(sequences):
    for ww, wave_idx in enumerate(seq):
        if wave_idx == eg_probe:
            x0 = ww - 1
            y0 = ss - egbox_height/2
            xd, yd = 2, egbox_height  # 2 seconds widht, 2*norm wave
            x = [x0, x0, x0+xd, x0+xd, x0]
            y = [y0, y0+yd, y0+yd, y0, y0]
            _ =  fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                          line=dict(color='black',
                                                    width=2,
                                                    dash='dot'),
                                          showlegend=False,
                                          )
                               )

# test show
_ = fig.update_xaxes(title_text='time (s)', title_standoff=0, range=[-0.1,6.1])
_ = fig.update_yaxes(tickmode='array',
                     tickvals=list(range(4)),
                     ticktext=[f'Seq.{i+1}' for i in range(4)],
                     ticks='',
                     showline=False)
fig.update_layout(width=96*3, height=96*1.5,
                  margin={'l':10,'r':10,'t':10,'b':10,},
                  template='simple_white')

# ensures transparent backgrounds
# fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
#                   plot_bgcolor='rgba(0,0,0,0)')

filename = folder / 'fig1_acquisition_order'
fig.write_image(filename.with_suffix('.png'))
fig.write_image(filename.with_suffix('.svg'))
fig.show()


### sounds organized by transition types
panelname = 'transitions'

fig = go.Figure()
all_figs.append(fig)
xbox = np.asarray([0, 1, 1, 0, 0])
ybox = (np.asarray([0, 0, 1, 1, 0]) - 0.5) * 0.75

for ww, (wave, color) in enumerate(zip(waves, colors)):
    # context box
    if ww > 0: # omits silence box,
        _ = fig.add_trace(go.Scatter(x=xbox - 1, y=ybox + ww, fill='toself',
                                     mode='lines',
                                     line=dict(width=1,
                                               color='gray'),
                                     fillcolor=color,
                                     showlegend=False)
                          )
    # contex wave
    x = np.linspace(-1, 0, n_samps)  # sum to offset to center, insline with sequences
    y = wave + ww
    _ = fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                 line=dict(color='black',
                                           width=1, ),
                                 showlegend=False)
                      )

    # probe box
    _ = fig.add_trace(go.Scatter(x=xbox, y=ybox + ww, fill='toself',
                                 mode='lines',
                                 line=dict(width=1,
                                           color='gray'),
                                 fillcolor=colors[eg_probe],
                                 showlegend=False)
                      )
    # probe wave
    x = np.linspace(0, 1, n_samps)
    y = waves[eg_probe] + ww
    _ = fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                 line=dict(color='black',
                                           width=1, ),
                                 showlegend=False)
                      )
    # ax.plot(x, y, colors[prb_idx])

    # context type text
    if ww == 0:
        type_text = 'silence'
    elif ww == eg_probe:
        type_text = 'same'
    else:
        type_text = 'different'

    _ = fig.add_trace(go.Scatter(x=[-1.1], y=[ww],
                                 mode='text', text=[type_text],
                                 textposition='middle left', textfont_size=11,
                                 showlegend=False)
                      )

_ = fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1,
                  )
# context and probe text
_ = fig.add_trace(go.Scatter(x=[-0.2, 0.2],
                             y=[-1, -1],
                             mode='text', text=['<b>Context</b>', '<b>Probe</b>'],
                             textposition=['middle left', 'middle right'], textfont_size=12,
                             showlegend=False)
                  )

# test show
_ = fig.update_layout(width=96 * 2.5, height=96 * 1.5,
                      margin={'l': 10, 'r': 10, 't': 10, 'b': 10, },
                      template='simple_white',
                      xaxis=dict(range=[-2,1.5], visible=False),
                      yaxis=dict(visible=False))

filename = folder / 'fig1_analysis_order'
fig.write_image(filename.with_suffix('.png'))
fig.write_image(filename.with_suffix('.svg'))
fig.show()


# plot both together to ensure shared x axis
fig = make_subplots(rows=1, cols=2, vertical_spacing=0.1, horizontal_spacing=0.05,)
# figure size in inches at different PPIs
ppi = 96  # www standard

heigh = 1.5
width = 5.5  # in inches
_ = fig.update_layout(template='simple_white',
                      margin=dict(l=10, r=10, t=10, b=10),
                      width=round(ppi * width), height=round(ppi * heigh),

                      # sequences
                      xaxis=dict(domain=[0, 0.55],
                                 autorange=True,
                                 constrain='range',
                                 tickmode='array',
                                 tickvals=[0, 1],
                                 ticktext=[0, 1],
                                 showline=False),
                      yaxis=dict(scaleanchor='y2',
                                 autorange=True,
                                 constrain='domain',
                                 tickmode='array',
                                 tickvals=list(range(4)),
                                 ticktext=[f'Seq.{i + 1}' for i in range(4)],
                                 ticks='',
                                 showline=False),

                      # transitions
                      xaxis2=dict(domain=[0.55, 1],
                                  constrain='range',
                                  scaleanchor='x',
                                  autorange=True, visible=False),
                      yaxis2=dict(visible=False),

                      showlegend=False,
                      font_size=10,
                      )
# top left
pan = all_figs[0]['data']
fig.add_traces(pan, cols=[1] * len(pan), rows=[1] * len(pan))

# top right
pan = all_figs[1]['data']
fig.add_traces(pan, cols=[2] * len(pan), rows=[1] * len(pan))

filename = folder / 'fig1_composite_order'
fig.write_image(filename.with_suffix('.png'))
fig.write_image(filename.with_suffix('.svg'))
fig.show()
