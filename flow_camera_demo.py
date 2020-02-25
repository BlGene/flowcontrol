from gym_grasping.flow_control.flow_module import FlowModule
from gym_grasping.robot_io.realsense2_cam import RealsenseCam
import numpy as np
import cv2
import matplotlib.pyplot as plt

from pdb import set_trace

if __name__ == "__main__":
    cam = RealsenseCam()


    prev_image = None

    prev_image, _ =  cam.get_image()
    print("before", prev_image.shape)
    new_size = tuple([int(x*0.5) for x in prev_image.shape[:2]])
    prev_image = np.array(cv2.resize(prev_image, new_size[::-1]))
    print("after", prev_image.shape)

    print(new_size)
    flow_module = FlowModule(size=new_size[::-1])

    for i in range(int(1e6)):
        image, _ = cam.get_image()
        new_size = tuple([int(x*0.5) for x in image.shape[:2]])
        image = np.array(cv2.resize(image, new_size[::-1]))
        #print(image.shape)

        flow = flow_module.step(prev_image,image)
        flow_img = flow_module.computeImg(flow, dynamic_range=False)

        images = np.hstack((prev_image,image, flow_img))
        # reise
        new_size = tuple([int(x*1.5) for x in images.shape[:2]])
        images = cv2.resize(images, new_size[::-1])
        # show
        cv2.imshow("rgb", images[:, :, ::-1])
        cv2.waitKey(1)

        #if i % 10 == 0:
        prev_image = image
        #  print("XXX")




