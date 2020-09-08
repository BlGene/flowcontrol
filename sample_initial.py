"""
Testing file for development, to experiment with evironments.
"""
import os
from PIL import Image
import cv2

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from gym_grasping.flow_control.flow_module_flownet2 import FlowModule


def sample_initial(sample_dir, task_name="stack", show=True, save=True):
    """
    Control the robot interactively by controlling sliders in the debug viewer.
    Be carefull not to dof sliders while over the robot becasue mouse actions
    go through panel.
    """
    quiver = True
    print([64*i for i in range(10)])
    env = RobotSimEnv(task=task_name, renderer='debug', act_type='continuous',
                      img_size=(256, 256),
                      max_steps=5.0)

    flow_module = FlowModule(size=env.img_size)

    done = False
    counter = 0
    prev_state = None  # will be set in loop
    for i in range(25):
        action = [0, 0, 0, 0, 1]
        state, reward, done, info = env.step(action)
        # print("reward:", reward)
        if done or reward > 0:
            raise ValueError

        if (show or save) and i > 0:
            flow = flow_module.step(prev_state, state)
            flow_img = flow_module.computeImg(flow, dynamic_range=False)

        if save:
            image_fn = os.path.join(sample_dir, f"sample_{i:03}.png")
            Image.fromarray(state).save(image_fn)

            if i > 0:
                flow_fn = os.path.join(sample_dir, f"sample_{i:03}_flow.png")
                Image.fromarray(flow_img).save(flow_fn)

        if show:
            img = state[:, :, ::-1]
            cv2.imshow('window', cv2.resize(img, (300, 300)))
            cv2.waitKey(1)

            if i > 0:
                if quiver:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(2, 2)
                    fig.tight_layout()
                    ax[0, 0].imshow(prev_state)
                    ax[0, 1].imshow(state)
                    ax[1, 0].imshow(flow_img)

                    f = 4
                    ax[1, 1].quiver(-flow_module.field[::f, ::f, 1],
                                    -flow_module.field[::f, ::f, 0],
                                    flow[::f, ::f, 0],
                                    -flow[::f, ::f, 1],
                                    angles='xy', scale_units='xy')
                    ax[1, 1].set_aspect(1.0)
                    for a in ax.flatten():
                        a.set_axis_off()
                    plt.show()

        if show or save:
            prev_state = state

        env.reset()
        counter += 1

    print(reward)


if __name__ == "__main__":
    sample_dir = "./samples"
    os.makedirs(sample_dir, exist_ok=True)
    sample_initial(sample_dir)
