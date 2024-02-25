# type: ignore
import matplotlib.pyplot as plt
from IPython.display import clear_output


def plot_partial(plot_fn, *plot_args):
    clear_output(wait=True)
    fig = plot_fn(*plot_args)
    plt.show(fig)
