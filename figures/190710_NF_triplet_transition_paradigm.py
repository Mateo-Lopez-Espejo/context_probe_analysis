import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import math
import pathlib as pl


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)



# generates two random signals eache one the sum of  some  sinusoinds
# sound 1
amplitude = 10
n_samp = 300
time1 = 1000
x1 = np.linspace(0, time1, n_samp)
s1 = np.sin(x1*1) * amplitude
s2 = np.sin(x1*1.5) * amplitude
s3 = np.sin(x1*2) * amplitude
noise = np.random.normal(0,2,n_samp)
y = s1 + s2 + s3 + noise
sound1 = gaussian_filter1d(y,0.1)

# sound 2
amplitude = 10
n_samp = 300
time2 = 300
x2 = np.linspace(0, time2, n_samp)
s1 = np.sin(x2*0.5) * amplitude
s2 = np.sin(x2*0.25) * amplitude
env = np.sin((x2*math.pi/290) + math.pi/3)
noise = np.random.normal(0,2,n_samp)
y = (s1 + s2) * env  + noise
sound2 = gaussian_filter1d(y,0.1)

# generates a block of data to imshow

sounds = (sound1, sound2)
maps = (plt.cm.winter, plt.cm.autumn)

wavelets = list()
w_cmaps = list()
w_blocks = list()
full_waves = list()
for ii, (sound, map) in enumerate(zip(sounds, maps)):
    block = np.expand_dims(np.linspace(0,10,300), 0)
    X = np.linspace(0,3,300)
    fig = plt.figure()
    top = plt.subplot2grid((2, 3), (0, 0), rowspan=1, colspan=3, fig=fig)

    top.imshow(block,  cmap=map,interpolation="bicubic",
                    origin='lower',extent=[0,3,np.min(sound1),np.max(sound1)],aspect="auto",
               vmin=0,vmax=10)
    top.plot(X, sound, color='black', linewidth=2)
    top.axis('off')

    set_size(15,5, ax=top)

    for ss, start in enumerate([0, 100, 200]):
        full_waves.append(sound)
        w_cmaps.append(map)
        sub = plt.subplot2grid((2, 3), (1, ss), rowspan=1, colspan=1, fig=fig)
        wblock = block[:, start:start+100]
        w_blocks.append(wblock)
        sub.imshow(wblock,  cmap=map,interpolation="bicubic",
                    origin='lower',extent=[0,1,np.min(sound),np.max(sound)],aspect="auto",
               vmin=0,vmax=10)
        wavelet = sound[start:start+100]
        wavelets.append(wavelet)
        sub.plot(X[0:100], wavelet, color='black', linewidth=2)
        set_size(5, 5, ax=sub)
        sub.axis('off')

    fig.set_size_inches(12, 3)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)

    root = pl.Path(f'/home/mateo/Pictures/DAC2')
    if not root.exists(): root.mkdir(parents=True, exist_ok=True)
    png = root.joinpath(f'sound{ii}').with_suffix('.png')
    fig.savefig(png, transparent=True, dpi=100)
    svg = png = root.joinpath(f'sound{ii}').with_suffix('.svg')
    fig.savefig(svg, transparent=True)



## figure with the reordered snippets

triplets = np.asarray([[5 , 6 , 2 , 3 , 5],
                       [6 , 5 , 3 , 2 , 6],
                       [2 , 4 , 5 , 4 , 6],
                       [3 , 1 , 2 , 1 , 3]])


fig = plt.figure()

for trial in range(triplets.shape[0]):

    # silence
    ax = plt.subplot2grid((4, 6), (trial, 0), rowspan=1, colspan=1, fig=fig)
    ax.plot(np.linspace(0,1,100), np.zeros(100), color='black', linewidth=2)
    ax.axis('off')

    for sound in range(triplets.shape[1]):

        ax = plt.subplot2grid((4, 6), (trial, sound+1), rowspan=1, colspan=1, fig=fig)

        wblock = w_blocks[triplets[trial,sound]-1]
        map = w_cmaps[triplets[trial,sound]-1]
        full_w = full_waves[triplets[trial,sound]-1]
        ax.imshow(wblock, cmap=map, interpolation="bicubic",
                   origin='lower', extent=[0, 1, np.min(full_w), np.max(full_w)], aspect="auto",
                   vmin=0, vmax=10)
        wavelet = wavelets[triplets[trial,sound]-1]
        ax.plot(np.linspace(0,1,100), wavelet, color='black', linewidth=2)
        ax.axis('off')

fig.set_size_inches(19, 5)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)

root = pl.Path(f'/home/mateo/Pictures/DAC2')
if not root.exists(): root.mkdir(parents=True, exist_ok=True)
png = root.joinpath('trials').with_suffix('.png')
fig.savefig(png, transparent=True, dpi=100)
svg = png = root.joinpath('trials').with_suffix('.svg')
fig.savefig(svg, transparent=True)

