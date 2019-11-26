import tensorflow as tf
import os

content_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Gwanghwamun_Plaza_-_Gwanghwamun_gate_-_Gyeongbokgung_Palace_2016_-_hschrijver.jpg/1280px-Gwanghwamun_Plaza_-_Gwanghwamun_gate_-_Gyeongbokgung_Palace_2016_-_hschrijver.jpg"
style_url = "https://cdn.britannica.com/78/43678-050-F4DC8D93/Starry-Night-canvas-Vincent-van-Gogh-New-1889.jpg"

class DataManager:

    def __init__(self):
        pass

    def download_image(self, file_name, url):
        """
        Take source for url and returns saved path where images are downloaded.
        """

        img_path = tf.keras.utils.get_file(file_name, url)

        return img_path

    def resize_image(self, image_path):
        if not os.path.exists(image_path):
            print("PATH is not available. Download image first.")

        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (224, 224))
        return img