import numpy as np
from ipywidgets import widgets, interact, Layout
import matplotlib.pyplot as plt

def plot_tcp_stationary(tcp_pos, video_recording):
    vel_vec = np.diff(tcp_pos, axis=0)
    vel_scl = np.linalg.norm(vel_vec, axis=1)
    val, label =  vel_scl, "velocity"
    fig, (ax, ax2) = plt.subplots(2, 1)
    fig.suptitle("TCP Stationary")
    line = ax.imshow(video_recording[0])
    ax.set_axis_off()
    line1 = ax2.plot(tcp_pos[:,0], label="x")
    line2 = ax2.plot(tcp_pos[:,1], label="y")
    line3 = ax2.plot(tcp_pos[:,2], label="z")
    ax2.set_ylabel("position (x,y,z)")
    ax2.set_xlabel("frame number")
    ax2r = ax2.twinx()
    ax2r.set_ylabel("velocity/threshold")
    line4 = ax2r.plot(val, label=label, color="b")
    #ax2r.axhline(y=vel_stable_threshold, linestyle="--", color="k")
    vline = ax2.axvline(x=2, color="k")
    lns = line1+line2+line3+line4
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs)
    #ax2.legend()
    max_frame = len(video_recording)

    def update(w):
        print("{} @ {} is {}".format(label, w, val[w] if w<max_frame else "?"))
        vline.set_data([w, w], [0, 1])
        line.set_data(video_recording[w])
        fig.canvas.draw_idle()

    slider_w = widgets.IntSlider(min=0, max=max_frame, step=1, value=0,
                                 layout=Layout(width='70%'))
    interact(update, w=slider_w)


def plot_gripper_stable(gripper_width, gripper_actions, grip_stable_arr, video_recording):
    fig, (ax, ax2) = plt.subplots(2, 1)
    fig.suptitle("Gripper Filter")
    line = ax.imshow(video_recording[0])
    ax.set_axis_off()
    line1 = ax2.plot((gripper_actions+1)/2,"--", label="gripper action", color="r")
    ax2.set_ylabel("grip stable")
    ax2.set_xlabel("frame number")
    ax2r = ax2.twinx()
    line3 = ax2r.plot(gripper_width, "--", label="gripper_widths", color="b")
    vline = ax2.axvline(x=2, color="k")
    line2 = ax2.plot(grip_stable_arr, label="grip stable", color="g")
    lns = line1+line2+line3
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs)

    def update(w):
        print("{} @ {} is {}".format("gripper_width", w, gripper_width[w]))
        vline.set_data([w, w], [0, 1])
        line.set_data(video_recording[w])
        fig.canvas.draw_idle()

    max_frame = len(gripper_width)-1
    slider_w = widgets.IntSlider(min=0, max=max_frame, step=1, value=0,
                                 layout=Layout(width='70%'))
    interact(update, w=slider_w)
