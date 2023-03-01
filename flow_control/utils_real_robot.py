import numpy as np


def move_to_neutral_safe(robot):
    home_pos = np.array([0.2988764, 0.11044048, 0.15792169])
    home_orn = np.array([9.99999242e-01, -1.23099822e-03, 1.18825773e-05, 3.06933556e-05])
    pos, orn = robot.get_tcp_pos_orn()

    if pos[2] > home_pos[2]:
        # current pos higher than home pos, add minimal delta_z
        delta_z = 0.03
    else:
        delta_z = abs(pos[2] - home_pos[2])

    pos_up = pos + np.array([0.0, 0.0, delta_z])
    pos_up2 = np.array([0.2988764, 0.11044048, pos_up[2]])

    robot.move_cart_pos(pos_up, orn, ref="abs", blocking=True, path="lin")
    robot.move_cart_pos(pos_up2, home_orn, ref="abs", blocking=True, path="lin")
    robot.move_cart_pos(home_pos, home_orn, ref="abs", blocking=True, path="lin")
