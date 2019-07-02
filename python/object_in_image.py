import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import os


BOX_BORDER_COLOR = {
    'Car': [1, 0, 0],               # red
    'DontCare': [0, 0, 0],          # black
    'Pedestrian': [0, 0, 1],        # blue
    'Van': [1, 1, 0],               # yellow
    'Cyclist': [1, 0, 1],           # magenta
    'Truck': [0, 1, 1],             # cyan
    'Misc': [0.5, 0, 0],            # maroon
    'Tram': [0, 0.5, 0],            # green
    'Person_sitting': [0, 0, 0.5]}  # navy

# border width
BOX_BORDER_WIDTH = 5


def show_object_in_image(img_filename, label_filename):
    img = mping.imread(img_filename)
    with open(label_filename) as f_label:
        lines = f_label.readlines()
        for line in lines:
            line = line.strip('\n').split()
            left_pixel, top_pixel, right_pixel, bottom_pixel = [int(float(line[i])) for i in range(4, 8)]
            box_border_color = BOX_BORDER_COLOR[line[0]]
            for i in range(BOX_BORDER_WIDTH):
                img[top_pixel+i, left_pixel:right_pixel, :] = box_border_color
                img[bottom_pixel-i, left_pixel:right_pixel, :] = box_border_color
                img[top_pixel:bottom_pixel, left_pixel+i, :] = box_border_color
                img[top_pixel:bottom_pixel, right_pixel-i, :] = box_border_color
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    
    IMG_DIR = '/home/dtc/Data/KITTI/data_object_image_2/training/image_2'
    LABEL_DIR = '/home/dtc/Data/KITTI/data_object_label_2/training/label_2'

    file_id = 0
    img_filename = os.path.join(IMG_DIR, '{0:06d}.png'.format(file_id))
    label_filename = os.path.join(LABEL_DIR, '{0:06d}.txt'.format(file_id))
    show_object_in_image(img_filename, label_filename)