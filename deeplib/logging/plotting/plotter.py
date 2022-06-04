import os
from numbers import Number

import plotext as plt

plt.theme("clear")


class Plotter:
    def __init__(self, total=100, min_plot_size=50):
        plt.xlim(0, total)
        plt.ylim(0, 1)

        t_size = os.get_terminal_size()
        plt.plot_size(t_size.columns, t_size.lines // 2)

        self.state = {}

        self.min_plot_size = min_plot_size

    def add_scalars(self, tag, dict):
        assert all(
            isinstance(v, Number) for v in dict.values()
        ), "Non-scalar value found."

        if tag not in self.state:
            self.state[tag] = {}

        for k, v in dict.items():
            if k not in self.state[tag]:
                self.state[tag][k] = []

            self.state[tag][k].append(v)

        self.refresh()

    def refresh(self):
        """
        Refresh plot with values in state.
        """

        num_plots = len(self.state)

        t_size = os.get_terminal_size()

        cols = t_size.columns // self.min_plot_size
        rows = t_size.lines // self.min_plot_size

        plt.cld()
        plt.clt()
        plt.subplots(None, cols)
        plt.plot_size(t_size.columns, t_size.lines // 2)

        for i, (tag, values) in enumerate(self.state.items()):
            plt.subplot(i // cols + 1, i % cols + 1).title(tag)

            for name, vals in values.items():
                plt.plot(vals, marker="hd")

        plt.show()
