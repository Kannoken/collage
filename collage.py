import cv2
import numpy as np
import sys
import tkinter
import urllib
import vk
import multiprocessing as mp
from functools import partial
import numpy as np

#load images from vk api
def load_images(count):
    session = vk.Session(access_token='TOKEN') #fill access_token
    vk_api = vk.API(session)
    image_data = vk_api.photos.getAll(v='5.0', album_id='wall', count=count)['items']
    return image_data

#get image by url from vk API
def from_url_to_img(url, x_size, y_size):
    resp = urllib.request.urlopen(url['photo_604'])
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (x_size, y_size))
    return image


#parse agruments from command line
def get_arg(arguments):
    d = dict(zip(arguments[::2], arguments[1::2]))
    return d.get('--count'), d.get('--size_cell'), d.get('--size_collage')


def make_collage(images, size_collage, size_cell):
    x, y = map(int, size_collage.split('x'))
    x_size, y_size = map(int, size_cell.split('x'))
    gor = []
    for i in range(y):
        vertical = []
        for j in range(x):
            if j + i * x < len(images):
                vertical.append(images[j + i * x])
            else:
                #add black square if image is empty
                vertical.append(np.zeros((x_size, y_size, 3), np.uint8))
        gor.append(np.hstack(vertical))
    collage = np.vstack(gor)
    cv2.imshow('collage', collage)
    cv2.waitKey(0)


if __name__ == "__main__":
    count, size_cell, size_collage = get_arg(sys.argv[1:])
    vk_data = load_images(count)
    pool = mp.Pool(mp.cpu_count())
    x_size, y_size = map(int, size_cell.split('x'))
    from_url = partial(from_url_to_img, x_size=x_size, y_size=y_size)
    images = pool.map(from_url, vk_data)
    make_collage(images, size_collage, size_cell)
    pool.close()
    pool.join()
