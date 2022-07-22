"""
Live plot of the servoing data.
"""
import os
import time
import logging
from collections import deque
from multiprocessing import Process, Pipe
import numpy as np
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button as Button
from scipy import ndimage

from flow_control.flow.flow_plot import FlowPlot


from pdb import set_trace


class ViewPlots(FlowPlot):
    """
    Live plot of the servoing data.
    """

    def __init__(self, size=(2, 1), threshold=None, save_dir=False):
        super().__init__()

        self.num_plots = 4
        self.image_size = (128, 128)
        self.horizon_timesteps = 50

        self.names = ["loss", "demo frame", "t", "fit q", "live z"]
        self.cur_plots = [None for _ in range(self.num_plots)]
        self.timesteps = 0
        self.data = [deque(maxlen=self.horizon_timesteps) for _ in range(self.num_plots)]

        self.save_dir = save_dir
        if save_dir:
            experiment_num = 0
            while os.path.isdir(f'{self.save_dir}_{experiment_num}'):
                experiment_num += 1

            self.save_dir = f'{self.save_dir}_{experiment_num}'
            os.makedirs(self.save_dir, exist_ok=False)

        plt.ion()
        self.fig = plt.figure(figsize=(8 * size[1], 3 * size[0]))
        g_s = gridspec.GridSpec(size[0], 3)
        g_s.update(wspace=0.001, hspace=.001)  # set the spacing between axes.
        plt.subplots_adjust(wspace=0.5, hspace=0, left=0, bottom=.05, right=1,
                            top=.95)

        # images stuff
        self.image_1_ax = plt.subplot(g_s[0, 0])
        self.image_2_ax = plt.subplot(g_s[0, 1])
        self.image_3_ax = plt.subplot(g_s[0, 2])
        self.image_1_ax.set_title("live state")
        self.image_2_ax.set_title("demo state")
        self.image_3_ax.set_title("flow")
        self.image_1_ax.set_axis_off()
        self.image_2_ax.set_axis_off()
        self.image_3_ax.set_axis_off()
        zero_image = np.zeros(self.image_size)
        self.image_plot_1_h = self.image_1_ax.imshow(zero_image)
        self.image_plot_2_h = self.image_2_ax.imshow(zero_image)
        self.image_plot_3_h = self.image_3_ax.imshow(zero_image)
        self.arrow_flow = self.image_3_ax.annotate("", xytext=(64, 64), xy=(84, 84),
                                                   arrowprops=dict(arrowstyle="->"))
        self.arrow_act = self.image_3_ax.annotate("", xytext=(64, 64), xy=(84, 84),
                                                  arrowprops=dict(arrowstyle="->"))
        #self.arrow_demo = self.image_3_ax.annotate("", xytext=(64, 64), xy=(84, 84),
        #                                          arrowprops=dict(arrowstyle="->"))
        #self.arrow_live = self.image_3_ax.annotate("", xytext=(64, 64), xy=(84, 84),
        #                                          arrowprops=dict(arrowstyle="->"))


        self.ax1 = plt.subplot(g_s[1, :])
        self.axes = [self.ax1, self.ax1.twinx(), self.ax1.twinx()]
        self.axes.append(self.axes[-1])

        if threshold is not None:
            self.axes[0].axhline(y=threshold, linestyle='dashed', color="k")

        def set_started_on(x):
            print(x)
            self.started = True

        self.started = False
        self.callback_s = set_started_on
        self.ax_start = plt.axes([0.04, 0.45, 0.25, 0.10])
        self.b_start = Button(self.ax_start, "Start")
        self.b_start.on_clicked(self.callback_s)

        plt.pause(1e-9)

    def __del__(self):
        plt.ioff()
        plt.close(self.fig)

    def reset(self):
        '''reset cached data'''
        self.timesteps = 0
        self.data = [deque(maxlen=self.horizon_timesteps) for _ in range(self.num_plots)]

    def step(self, series_data, live_rgb, demo_rgb, flow, demo_mask,
             action):
        """
        Step the plotting.

        Arguments:
            series_data: list of (loss, base_frame, val_1, ...)
            live_rgb: image with shape (h, w, 3)
            demo_rgb: image with shape (h, w, 3)
            flow: image with shape (h, w, 2)
            demo_mask: None or image with shape (h, w)
            action: None or dict(motion=..., ref=)
        """
        assert isinstance(action, dict)

        # 0. compute flow image
        flow_img = self.compute_image(flow)

        if demo_mask is not None and np.any(demo_mask):
            # 1. edge around object
            edge = np.gradient(demo_mask.astype(float))
            edge = (np.abs(edge[0]) + np.abs(edge[1])) > 0
            flow_img[edge] = (255, 0, 0)

            # 2. compute mean flow direction
            mean_flow = np.mean(flow[demo_mask], axis=0)
            # mean_flow = np.clip(mean_flow*flow_s, -63, 63)  # clip to assure plot

            # plot from center of image
            # mean_flow_origin = [int(round(x/2)) for x in self.image_size]
            # mean_flow2 = mean_flow / np.linalg.norm(mean_flow)
            # mean_flow_xy = mean_flow_origin + mean_flow2 * (self.image_size[0]/2-1)

            # plot from center of segmentation
            mask_com = np.array(ndimage.center_of_mass(demo_mask))[::-1]
            size_scl = np.array(self.image_size) /  demo_mask.shape
            mean_flow_origin = mask_com * size_scl
            mean_flow_xy = mean_flow_origin + mean_flow * size_scl

            if np.any(mean_flow_xy > self.image_size):
                logging.warning("Not showing mean flow arrow.")

            # TODO(max), clip this to image, via scaling.
            self.arrow_flow.remove()
            del self.arrow_flow
            arrw_f = self.image_3_ax.annotate("", xytext=mean_flow_origin,
                                              xy=mean_flow_xy,
                                              arrowprops=dict(arrowstyle="->"))
            self.arrow_flow = arrw_f

        # update images
        self.image_plot_1_h.set_data(live_rgb)
        self.image_plot_2_h.set_data(demo_rgb)
        self.image_plot_3_h.set_data(flow_img)

        if self.arrow_act:
            self.arrow_act.remove()
            del self.arrow_act
            self.arrow_act = None

        if action is not None:
            act_s = 1e2
            # act_in_img = action[0:2] / np.linalg.norm(action[0:2]) * 63
            x, y = action["motion"][0][0:2]
            act_in_img = np.clip((x * act_s, y * act_s), -63, 63)
            act_in_img = (64 - act_in_img[0], 64 + act_in_img[1])
            arrw_a = self.image_3_ax.annotate("", xytext=(64, 64),
                                              xy=act_in_img,
                                              arrowprops=dict(arrowstyle="->",
                                                              color='b'))
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
            name = self.names[i]
            col = "k" if name == "t" else 'C{}'.format(i)
            res = self.axes[i].plot(range(xmin, xmax), list(self.data[i]),
                                    color=col, label=name)
            # self.axes[i].relim()
            # self.axes[i].autoscale_view()
            self.cur_plots[i], = res
            self.ax1.set_xlim(xmin, xmax)

        self.ax1.legend(handles=self.cur_plots, loc='upper center')

        # maybe see: https://bastibe.de/2013-05-30-speeding-up-matplotlib.html
        # was plt.pause(1e-9), but thi sis not needed anymore
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # flush before save.
        if self.save_dir:
            plot_fn = os.path.join(self.save_dir, "plot_{0:04}.jpg".format(self.timesteps))
            plt.savefig(plot_fn)


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

    def __init__(self, *args, **kwargs):
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

    def __del__(self):
        self.close()

    def step(self, *obs, **kwargs):
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
        view_plots.step(series_data, live_rgb, demo_rgb, flow_img, demo_mask=None, action=None)

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
        view_plots.step(series_data, live_rgb, demo_rgb, flow_img, None, None)

        time.sleep(.2)


if __name__ == "__main__":
    # test_normal()
    test_subproc()
