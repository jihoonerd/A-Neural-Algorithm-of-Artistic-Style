import tensorflow as tf
import os


def download_image(file_name, url):
    """
    Take source for url and returns saved path where images are downloaded.
    """

    img_path = tf.keras.utils.get_file(file_name, url)

    return img_path

def resize_image(image_path):
    if not os.path.exists(image_path):
        print("PATH is not available. Download image first.")

    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (224, 224))
    img = img[tf.newaxis, :]
    return img