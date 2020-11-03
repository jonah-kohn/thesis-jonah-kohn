import os
import matplotlib
from matplotlib import pyplot as plt
from skimage.draw import random_shapes

save_dir = os.path.join(os.getcwd(), "shape_data")
rect_dir = os.path.join(save_dir, "rectangles")
tri_dir = os.path.join(save_dir, "triangles")
# os.mkdir(rect_dir)
# os.mkdir(tri_dir)


def create_rectangles(n):
    for i in range(n):

        result, _ = random_shapes((256, 256), max_shapes=1, shape='rectangle',
                               min_size=50, multichannel=True)
        matplotlib.image.imsave(os.path.join(save_dir, "rectangles", str(i)+".jpg"), result)


def create_triangles(n):
    for i in range(n):

        result, _ = random_shapes((256, 256), max_shapes=1, shape='triangle',
                               min_size=50, multichannel=True)
        matplotlib.image.imsave(os.path.join(save_dir, "triangles", str(i)+".jpg"), result)


create_triangles(300)
create_rectangles(300)
