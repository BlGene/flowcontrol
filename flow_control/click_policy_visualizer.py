import numpy as np
import matplotlib
matplotlib.use("TkAgg")

from matplotlib import pyplot as plt
import cv2
from flow_control.demo_segment_util import transform_depth

T_TCP_CAM = np.array([[9.99801453e-01, -1.81777984e-02, 8.16224931e-03, 2.77370419e-03],
                      [1.99114100e-02, 9.27190979e-01, -3.74059384e-01, 1.31238638e-01],
                      [-7.68387855e-04, 3.74147637e-01, 9.27368835e-01, -2.00077483e-01],
                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

T_TCP_CAM = np.array([
   [0.99987185, -0.00306941, -0.01571176, 0.00169436],
   [-0.00515523, 0.86743151, -0.49752989, 0.11860651],
   [0.015156,    0.49754713,  0.86730453, -0.18967231],
   [0., 0., 0., 1.]])



class WaypointSelector:

    def __init__(self, rgb, depth):


        self.calib = {"width": 640, "height": 480, "fx": 616.1953125, "fy": 616.1953735351562, "ppx": 311.5280456542969, "ppy": 235.69309997558594}
        self.flat_depth = transform_depth(
            depth.copy(),
            np.linalg.inv(T_TCP_CAM),
            self.calib
        )
        self.depth = depth
        self.rgb = rgb

    def select_next(self):
        fig = plt.gcf()
        fig.canvas.mpl_connect('button_press_event', self.click_event)

        plt.imshow(self.rgb)
        plt.show()

        #self.click_event()

    def click_event(self, event):
        if event.xdata is None or event.ydata is None:
            return

        click = (int(event.xdata), int(event.ydata))
        # Compute mask according to surface
        target_depth = self.flat_depth[click[::-1]]
        target_mask = np.abs(self.flat_depth - target_depth) < 0.004
        plt.close()

        
        # Extract mask contours and get the one clicked
        contour = next(
            c for c in cv2.findContours(
                np.uint8(target_mask),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )[0] if cv2.pointPolygonTest(c, click, False) > 0
        )
        
        # Create mask from selected contour
        mask = np.zeros_like(self.rgb, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, (255, 0, 0), 1)
        cv2.fillPoly(mask, pts=[contour], color=(255, 0, 0))


        plt.imshow(cv2.addWeighted(self.rgb, 0.5, mask, 1.0 - 0.5, 0.0))
        plt.show()
        mask = mask[..., 0]

        img = self.rgb.copy()

        corners = cv2.goodFeaturesToTrack(mask, 4, 0.01, 10)
        corners = np.int0(corners)

        # xs = []
        # ys = []

        # for i in corners:
        #     x, y = i.ravel()
        #     xs += [x]
        #     ys += [y]
        #     cv2.circle(img, (x, y), 3, 255, -1)

        # cv2.circle(img, (int(np.mean(xs)), int(np.mean(ys))), 3, (0, 255, 0), -1)
        # print(int(np.mean(xs)), int(np.mean(ys)))

        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        print((int(cX), int(cY)))
        cv2.circle(img, (int(cX), int(cY)), 7, (255, 0, 0), -1)
        plt.imshow(img)
        plt.show()


        arclen = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, arclen*0.05, True)
        #drawContours
        cv2.drawContours(img, [approx], -1, (0,0,255), 1, cv2.LINE_AA)
        plt.imshow(img)
        plt.show()
        print("0")




        gray = cv2.cvtColor(self.rgb.copy(), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.Canny(gray, 50, 100)

        plt.imshow(gray)
        plt.show()
        print("1")

        # mask = cv2.dilate(mask, np.ones((11, 11), np.uint8))

        # gray = np.uint8(gray * (mask > 0))
        # gray = self.flat_depth.copy() #* (mask > 0)

        # plt.imshow(gray)
        # plt.show()

        # dy, dx = np.gradient(gray)
        # grad = np.square(dy * dy + dx * dx)
        # plt.imshow(np.log10(grad * (grad < 1e-7) + 1e-15))
        # plt.show()

        edges = self.flat_depth.copy() #target_depth * mask
        edges = (edges - edges.min()) / (edges.max() - edges.min())
        edges = np.uint8(edges * 255)
        
        #edges = cv2.dilate(edges, None)
        #edges = cv2.GaussianBlur(edges, (3, 3), 0)
        edges = cv2.Canny(edges, 100, 300)


        #plt.imshow(edges)
        #plt.show()

        # for i in corners:
        #     x, y = i.ravel()
        #     cv2.circle(edges, (x, y), 3, 255, -1)

        lines = cv2.HoughLinesP(
            gray, 1, np.pi / 180, 30
        )


        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)


        #plt.imshow(img)
        #plt.show()


if __name__ == "__main__":
    import os
    demo_dir = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/vacuum/"
    recording_dict = np.load(os.path.join(demo_dir, 'episode_5.npz'))

    print(list(recording_dict.keys()))

    video_recording = recording_dict["rgb_unscaled"]
    depth_recording = recording_dict['depth_imgs']

    print(video_recording.shape)
    print(depth_recording.shape)

    for i in range(1, 100, 4):
        click_recorder = WaypointSelector(video_recording[i], depth_recording[i])
        click_recorder.select_next()
