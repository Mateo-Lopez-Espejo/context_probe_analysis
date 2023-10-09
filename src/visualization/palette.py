from webcolors import hex_to_rgb

# tablau
Blue = '#4E79A7'
Orange = '#F28E2B'
Red = '#E15759'
Teal = '#76B7B2'
Green = '#59A147'
Yellow = '#EDC948'
Purple = '#B07AA1'
Pink = '#FF9DA7'
Brown = '#9C755F'
Grey = '#BAB0AC'

# extra colors
Black = '#000000'
DarkGray = '#A9A9A9'
SteelBlue = '#4682B4'
HardBlue = '#0000FF'

ColorDict = {
    'Blue': Blue, 'Green': Green, 'Orange': Orange, 'Yellow': Yellow,
    'Purple': Purple, 'Red': Red, 'Pink': Pink, 'Grey': Grey, 'Teal': Teal,
    'Brown': Brown
}
ColorList = list(ColorDict.values())

# global color parameters for CPN relevant categories

# 4 sound subset
FOURCOLOR = [Grey, Yellow, Red, Teal, Brown]
TENCOLOR = ColorList

# region
A1_COLOR = Blue
PEG_COLOR = Orange
REGION_COLORMAP = {'A1': A1_COLOR, 'PEG': PEG_COLOR}

# photo activated blue
PHOTOACT = HardBlue
NAROWSPIKE = DarkGray
BROADSPIKE = Black
CELLTYPE_COLORMAP = {
    'activated': PHOTOACT, 'narrow': NAROWSPIKE, 'broad': BROADSPIKE,
    'unclass': Yellow
}

# metric
AMPCOLOR = Green
DURCOLOR = Purple

# model colormap
MODEL_COLORMAP = {'matchl_STRF': Orange,
                  'matchl_self': Blue,
                  'matchl_pop': Green,
                  'matchl_full': Purple}


def add_opacity(hex, opacity):
    return f'rgba{(*hex_to_rgb(hex), opacity)}'
