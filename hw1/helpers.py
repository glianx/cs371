from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size


def fixed_axis_figure(box, scale=1., axis='on', h_pads=None, v_pads=None):
    if h_pads is None:
        h_pads = [0.7, 0.1] if axis == 'on' else [0, 0]
    if v_pads is None:
        v_pads = [0.5, 0.1] if axis == 'on' else [0, 0]
    ax_h_size, ax_v_size = scale * (box[1] - box[0]), scale * (box[3] - box[2])
    fig_h_size, fig_v_size = h_pads[0] + ax_h_size + h_pads[1], v_pads[0] + ax_v_size + v_pads[1]
    fig = plt.figure(figsize=(fig_h_size, fig_v_size))
    horizontal = [Size.Fixed(h_pads[0]), Size.Fixed(ax_h_size), Size.Fixed(h_pads[1])]
    vertical = [Size.Fixed(v_pads[0]), Size.Fixed(ax_v_size), Size.Fixed(v_pads[1])]
    d = Divider(fig, (0, 0, 1, 1), horizontal, vertical,
                aspect=False)
    ax = fig.add_axes(d.get_position(), axes_locator=d.new_locator(nx=1, ny=1))
    plt.xlim(box[:2])
    plt.ylim(box[2:])
    plt.axis(axis)
    return fig, ax


def make_figure(box, max_width=5, max_height=5):
    scales = [
        max_width / (box[1] - box[0]),
        max_height / (box[3] - box[2])
    ]
    scale = min(scales)
    fixed_axis_figure(box, scale=scale, axis='off')


ten_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:10]