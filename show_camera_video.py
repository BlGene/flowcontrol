"""
This reads the camera image, and displays it.
"""
import numpy as np
import cv2
from gym_grasping.flow_control.flow_module_flownet2 import FlowModule
from robot_io.cams.realsenseSR300_librs2 import RealsenseSR300


def main():
    '''This reads the camera image, and displays it.'''

    cam = RealsenseSR300()
    prev_image = None
    prev_image, _ = cam.get_image()
    print("before", prev_image.shape)
    new_size = tuple([int(x*0.5) for x in prev_image.shape[:2]])
    prev_image = np.array(cv2.resize(prev_image, new_size[::-1]))
    print("after", prev_image.shape)
    print(new_size)

    flow_module = FlowModule(size=new_size[::-1])
    for _ in range(int(1e6)):
        image, _ = cam.get_image()
        new_size = tuple([int(x*0.5) for x in image.shape[:2]])
        image = np.array(cv2.resize(image, new_size[::-1]))

        flow = flow_module.step(prev_image, image)
        flow_img = flow_module.computeImg(flow, dynamic_range=False)

        images = np.hstack((prev_image, image, flow_img))
        # reise
        new_size = tuple([int(x*1.5) for x in images.shape[:2]])
        images = cv2.resize(images, new_size[::-1])
        # show
        cv2.imshow("rgb", images[:, :, ::-1])
        cv2.waitKey(1)

        # if i % 10 == 0:
        prev_image = image


if __name__ == "__main__":
    main()
