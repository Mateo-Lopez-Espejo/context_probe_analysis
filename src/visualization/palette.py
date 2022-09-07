from webcolors import hex_to_rgb


# # plos recomendation
# Blue =  '#90CAF9'
# Green = '#C5E1A5'
# Orange = '#FFB74D'
# Yellow = '#FFF176'
# Purple = '#9E86C9'
# Red = '#E57373'
# Pink = '#F48FB1'
# Grey = '#E6E6E6'

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

ColorDict = dict(Blue=Blue, Green=Green, Orange=Orange, Yellow=Yellow, Purple=Purple, Red=Red, Pink=Pink, Grey=Grey,
                  Teal=Teal, Brown=Brown)
ColorList = list(ColorDict.values())

# global color parameters for CPN relevant categories

# 4 sound subset
FOURCOLOR = [Grey, Yellow, Red, Teal, Brown]
TENCOLOR = ColorList

# region
A1_COLOR = Blue
PEG_COLOR = Orange
REGION_COLORMAP = {'A1': A1_COLOR, 'PEG': PEG_COLOR}

# metric
AMPCOLOR = Green
DURCOLOR = Purple

# model colormap
MODEL_COLORMAP = {'matchl_STRF': Orange,
                  'matchl_self': Blue,
                  'matchl_pop': Green,
                  'matchl_full': Purple}

def add_opacity(hex, opacity):
     return  f'rgba{(*hex_to_rgb(hex), opacity)}'

if __name__ == "__main__":
    colors = dict(Blue=Blue, Green=Green, Orange=Orange, Yellow=Yellow, Purple=Purple, Red=Red, Pink=Pink, Grey=Grey,
                  Teal=Teal, Brown=Brown)
    import plotly.graph_objects as go
    fig = go.Figure()
    _ = fig.add_trace(go.Scatter(
        x=list(range(len(colors))),
        y=[1]*len(colors),
        mode='markers',
        marker=dict(color=list(colors.values()),
                    size=50,
                    opacity=1),
        text=[f'{key}: {val}' for key, val in colors.items()]))
    fig.show()
