import tensorflow as tf

content_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Gwanghwamun_Plaza_-_Gwanghwamun_gate_-_Gyeongbokgung_Palace_2016_-_hschrijver.jpg/1280px-Gwanghwamun_Plaza_-_Gwanghwamun_gate_-_Gyeongbokgung_Palace_2016_-_hschrijver.jpg"
style_url = "https://cdn.britannica.com/78/43678-050-F4DC8D93/Starry-Night-canvas-Vincent-van-Gogh-New-1889.jpg"



def download_image(content_url, style_url):
    """Take source for url and returns saved path where images are downloaded.
    
    Args:
        content_url (str): URL
        style_url   (str): URL

    Returns:
        content_img (str): PATH
        style_img   (str): PATH
    """

    content_path = tf.keras.utils.get_file("content.jpg", content_url)
    style_path = tf.keras.utils.get_file("style.jpg", style_url)

    return content_path, style_path


def load_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224)) # Input size of VGG array




if __name__ == "__main__":
    content_img, style_img = download_image(content_url, style_url)
    load_img(content_img)

