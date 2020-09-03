"""
Live plot of the servoing data.
"""
import time
from collections import deque
from multiprocessing import Process, Pipe

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from gym_grasping.flow_control.flow_plot import FlowPlot


class ViewPlots(FlowPlot):
    """
    Live plot of the servoing data.
    """

    def __init__(self, size=(2, 1), threshold=.1):
        super().__init__()

        plt.ion()
        self.fig = plt.figure(figsize=(8*size[1], 3*size[0]))
        g_s = gridspec.GridSpec(size[0], 3)
        g_s.update(wspace=0.001, hspace=.3)  # set the spacing between axes.
        plt.subplots_adjust(wspace=0.5, hspace=0, left=0, bottom=.05, right=1,
                            top=.95)

        self.num_plots = 2
        self.horizon_timesteps = 50
        self.ax1 = plt.subplot(g_s[1, :])
        self.image_plot_1 = plt.subplot(g_s[0, 0])
        self.image_plot_2 = plt.subplot(g_s[0, 1])
        self.image_plot_3 = plt.subplot(g_s[0, 2])

        self.axes = [self.ax1, self.ax1.twinx(), self.ax1.twinx()]
        self.axes.append(self.axes[-1])

        self.cur_plots = [None for _ in range(self.num_plots)]
        self.timesteps = 0
        self.data = [deque(maxlen=self.horizon_timesteps) for _ in range(self.num_plots)]

        self.names = ["loss", "demo frame", "demo z", "live z"]
        self.axes[0].axhline(y=threshold, color="k")

        # images stuff
        self.image_size = (128, 128)
        zero_image = np.zeros(self.image_size)
        self.image_plot_1_h = self.image_plot_1.imshow(zero_image)
        self.image_plot_1.set_axis_off()
        self.image_plot_1.set_title("live state")
        self.image_plot_2_h = self.image_plot_2.imshow(zero_image)
        self.image_plot_2.set_axis_off()
        self.image_plot_2.set_title("demo state")
        self.image_plot_3_h = self.image_plot_3.imshow(zero_image)
        self.image_plot_3.set_axis_off()
        self.image_plot_3.set_title("flow")

        arrow_flow = self.image_plot_3.annotate("", xytext=(64, 64),
                                                xy=(84, 84),
                                                arrowprops=dict(arrowstyle="->")
                                                )
        arrow_act = self.image_plot_3.annotate("", xytext=(64, 64),
                                               xy=(84, 84),
                                               arrowprops=dict(arrowstyle="->")
                                               )
        self.arrow_flow = arrow_flow
        self.arrow_act = arrow_act

        # plt.show(block=False)
        plt.show()

    def __del__(self):
        plt.ioff()
        plt.close()

    def reset(self):
        '''reset cached data'''
        self.timesteps = 0
        self.data = [deque(maxlen=self.horizon_timesteps) for _ in range(self.num_plots)]

    def step(self, series_data, live_rgb, demo_rgb, flow, demo_mask, action):
        '''step the plotting'''

        # 0. compute flow image
        flow_img = self.compute_image(flow)

        # 1. edge around object
        edge = np.gradient(demo_mask.astype(float))
        edge = (np.abs(edge[0]) + np.abs(edge[1])) > 0
        flow_img[edge] = (255, 0, 0)

        # 2. compute mean flow direction
        mean_flow = np.mean(flow[demo_mask], axis=0)
        mean_flow_xy = (64+mean_flow[0], 64+mean_flow[1])
        self.arrow_flow.remove()
        del self.arrow_flow
        arrw_f = self.image_plot_3.annotate("", xytext=(64, 64),
                                            xy=mean_flow_xy,
                                            arrowprops=dict(arrowstyle="->"))
        self.arrow_flow = arrw_f

        self.arrow_act.remove()
        del self.arrow_act
        act_in_img = (64 + action[0]*1e3, 64 + action[1]*1e3)
        arrw_a = self.image_plot_3.annotate("", xytext=(64, 64),
                                            xy=act_in_img,
                                            arrowprops=dict(arrowstyle="->"),
                                            color='blue')
        self.arrow_act = arrw_a

        for point, series in zip(series_data, self.data):
            series.append(point)
        self.timesteps += 1
        xmin = max(0, self.timesteps - self.horizon_timesteps)
        xmax = self.timesteps
        for plot in self.cur_plots:
            if plot is not None:
                plot.remove()

        for i in range(self.num_plots):
            col = 'C{}'.format(i)
            lbls = self.names[i]
            res = self.axes[i].plot(range(xmin, xmax), list(self.data[i]),
                                    color=col, label=lbls)
            self.cur_plots[i], = res
            self.ax1.set_xlim(xmin, xmax)

        self.ax1.legend(handles=self.cur_plots, loc='upper center')

        # next do images
        self.image_plot_1_h.set_data(live_rgb)
        self.image_plot_2_h.set_data(demo_rgb)
        self.image_plot_3_h.set_data(flow_img)

        # flush
        self.fig.tight_layout()
        self.fig.canvas.draw()

        # TOO
        save_plots = False
        if save_plots:
            plot_name = "./save_plots_wheel/img_{0:03}".format(self.timesteps)
            plt.savefig(plot_name)

        # pause not needed
        plt.pause(0.001)


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                env.step(*data)
            elif cmd == 'reset':
                env.reset()
            elif cmd == 'close':
                # del env
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocPlot worker: got KeyboardInterrupt')
    finally:
        # env.close()
        del env


class SubprocPlot():
    """
    Wrap the plotting in a subprocess so that we don't get library import
    collisions for Qt/OpenCV, e.g. with RLBench
    """

    def __init__(self):
        self.waiting = False
        self.closed = False
        subproc_class = ViewPlots

        num_plots = 1
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_plots)])
        self.remotes = list(self.remotes)
        self.work_remotes = list(self.work_remotes)
        self.ps = [Process(target=worker,
                           args=(self.work_remotes[0], self.remotes[0],
                                 subproc_class))]
        # for (work_remote, remote, sp_class) in zip(self.work_remotes,
        # self.remotes, [subproc_class])]
        for p in self.ps:
            # if the main process crashes, we should not cause things to hang
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, *obs):
        self._assert_not_closed()
        for remote, observation in zip(self.remotes, [obs]):
            remote.send(('step', observation))
        # self.waiting = True

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        # remotes_responsive = [remote.poll(10) for remote in self.remotes]
        # while not np.all(remotes_responsive):
        #    print(remotes_responsive)
        #    print("restart envs")
        #    raise ValueError
        #     self.restart_envs(remotes_responsive)
        #     remotes_responsive = [remote.poll(10) for remote in self.remotes]
        return

    def close(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a Subproc after"
        " calling close()"


def test_normal():
    base = np.ones((128, 128, 3), dtype=np.uint8)
    live_rgb = base * 0
    demo_rgb = base * 255
    flow_img = base * 128
    view_plots = ViewPlots()

    loss = .5
    demo_frame = 1
    ee_pos = base_pos = [0, 0, 0]

    for iter in range(10):
        print("iter", iter)
        demo_frame += 1
        loss *= 0.9
        series_data = (loss, demo_frame, base_pos[0], ee_pos[0])
        view_plots.step(series_data, live_rgb, demo_rgb, flow_img)

        time.sleep(.2)


def test_subproc():
    base = np.ones((128, 128, 3), dtype=np.uint8)
    live_rgb = base * 0
    demo_rgb = base * 255
    flow_img = base * 128
    view_plots = SubprocPlot()

    loss = .5
    demo_frame = 1
    ee_pos = base_pos = [0, 0, 0]

    for iter in range(10):
        print("iter", iter)
        demo_frame += 1
        loss *= 0.9
        series_data = (loss, demo_frame, base_pos[0], ee_pos[0])
        view_plots.step(series_data, live_rgb, demo_rgb, flow_img)

        time.sleep(.2)

    view_plots.close()


if __name__ == "__main__":
    # test_normal()
    test_subproc()
