"""
Live plot of the servoing data.
"""
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class ViewPlots:
    """
    Live plot of the servoing data.
    """
    def __init__(self, size=(2, 1), threshold=.1):
        plt.ion()
        self.fig = plt.figure(figsize=(8*size[1], 3*size[0]))
        g_s = gridspec.GridSpec(size[0], 3)
        g_s.update(wspace=0.001, hspace=.3) # set the spacing between axes.
        plt.subplots_adjust(wspace=0.5, hspace=0, left=0, bottom=.05, right=1, top=.95)

        self.num_plots = 4
        self.horizon_timesteps = 50
        self.ax1 = plt.subplot(g_s[1, :])
        self.low_1 = plt.subplot(g_s[0, 0])
        self.low_2 = plt.subplot(g_s[0, 1])
        self.low_3 = plt.subplot(g_s[0, 2])

        self.axes = [self.ax1, self.ax1.twinx(), self.ax1.twinx()]
        self.axes.append(self.ax[-1])

        self.cur_plots = [None for _ in range(self.num_plots)]
        self.timesteps = 0
        self.data = [deque(maxlen=self.horizon_timesteps) for _ in range(self.num_plots)]

        self.names = ["loss", "demo frame", "demo z", "live z"]
        self.axes[0].axhline(y=threshold, color="k")

        # images stuff
        self.low_1_h = self.low_1.imshow(np.zeros((256, 256)))
        self.low_1.set_axis_off()
        self.low_1.set_title("live state")
        self.low_2_h = self.low_2.imshow(np.zeros((256, 256)))
        self.low_2.set_axis_off()
        self.low_2.set_title("demo state")
        self.low_3_h = self.low_3.imshow(np.zeros((256, 256)))
        self.low_3.set_axis_off()
        self.low_3.set_title("flow")
        #plt.show(block=False)
        plt.show()

    def __del__(self):
        plt.ioff()
        plt.close()

    def reset(self):
        '''reset cached data'''
        self.timesteps = 0
        self.data = [deque(maxlen=self.horizon_timesteps) for _ in range(self.num_plots)]


    def step(self, *obs):
        '''step the plotting'''
        for point, series in zip(obs, self.data):
            series.append(point)

        self.timesteps += 1
        xmin, xmax = max(0, self.timesteps - self.horizon_timesteps), self.timesteps

        for plot in self.cur_plots:
            if plot is not None:
                plot.remove()

        for i in range(self.num_plots):
            col = 'C{}'.format(i)
            lbls = self.names[i]
            res = self.axes[i].plot(range(xmin, xmax), list(self.data[i]), color=col, label=lbls)
            self.cur_plots[i], = res
            self.ax1.set_xlim(xmin, xmax)

        self.ax1.legend(handles=self.cur_plots, loc='upper center')
        self.fig.tight_layout()
        self.fig.canvas.draw()

        save_plots = False
        if save_plots:
            plot_name = "./save_plots_wheel/img_{0:03}".format(self.timesteps)
            plt.savefig(plot_name)

        # pause not needed
        # plt.pause(0.001)
