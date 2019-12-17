import os
import urllib.request

import tensorflow as tf


def download_image(file_name, url):
    """
    Take source for url and returns saved path where images are downloaded.
    """

    if not os.path.exists("./source_images"):
        os.mkdir("source_images")
    file_path = "./source_images/" + file_name
    urllib.request.urlretrieve(url, file_path)
    return file_path

def resize_image(image_path):
    if not os.path.exists(image_path):
        print("PATH is not available. Download image first.")

    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (224, 224))
    img = img[tf.newaxis, :]
    return img
