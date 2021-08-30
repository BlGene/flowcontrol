import cv2
import numpy as np
import time
from copy import copy

from pdb import set_trace

def sample_line(arr, center, axis='x', width=64, height=1, offset=0, rev=False):
    """
    sample a line along either the x or y axis
    """
    x, y = center
    hw = int(width/2)
    hh = int(height/2)
    if axis == 'x':
        if height == 1:
            y_slice = slice(y, y+1)
        else:
            y_slice = slice(y-hh, y+hh+1)
        x_slice = slice(x-hw+offset, x+hw+offset+1)

    if axis == 'y':
        if height == 1:
            x_slice = slice(x, x+1)
        else:
            x_slice = slice(x-hh, x+hh+1)
        y_slice = slice(y-hw+offset, y+hw+offset+1)

    tmp = arr[y_slice, x_slice]
    if axis == 'y':
        tmp = tmp.transpose(1, 0)
    if rev:
        tmp = tmp[:, ::-1]
    return tmp

def center_axis(mag, clicked_point, axis):
    offsets = range(-8, 8)
    orig = sample_line(mag, clicked_point, axis=axis)
    samples = [sample_line(mag, clicked_point, offset=o, rev=True, axis=axis) for o in offsets]
    scores = [np.sum(s*orig) for s in samples]
    # print(scores/max(scores))
    am = np.argmax(scores)
    max_score = offsets[am]

    plot = False
    if plot:
        print("max_score", max_score)
        tmp = np.concatenate((orig, *samples), axis=0)
        tmp = (tmp - tmp.min()) / (tmp.max()-tmp.min())
        cv2.imshow("image", tmp)
        cv2.waitKey(1)
        time.sleep(3)

    return round(max_score/2)  # TODO(max): why is this needed?


def compute_center(rgb, depth, orig_clicked_point):
    img_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    #img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    #sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    center = copy(orig_clicked_point)
    for i in range(5):
        new_center = (center[0] + center_axis(mag, center, axis='x'),
                      center[1] + center_axis(mag, center, axis='y'))

        if new_center == center:
            break
        center = new_center

    return new_center


def main():
    from pdb import set_trace

    arr = np.zeros((100, 100, 3),dtype=np.uint8)
    center = (46, 43)
    size = 20
    arr[center[1]-(size-1)//2:center[1]+(size-1)//2+1,
        center[0]-(size-1)//2:center[0]+(size-1)//2+1] = 255

    # center of mass
    com = np.array(np.where(arr[:,:,0])).mean(axis=1)
    print("com", com[::-1])

    new_center = compute_center(arr, None, (42, 44))
    print("new_center", new_center)

    #cv2.imshow("image", arr)
    #cv2.waitKey(1)
    #time.sleep(10)

if __name__ == "__main__":
    main()


