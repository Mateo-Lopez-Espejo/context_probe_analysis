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
