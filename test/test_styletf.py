import os
import tempfile

import tensorflow as tf

from styletf.data.data_manager import DataProcessor
from styletf.misc.sample_images import content_url, style_url


def test_resize_img():

    with tempfile.TemporaryDirectory() as tmpdir:
        processor = DataProcessor()
        file_name = tmpdir + "/content.jpg"
        path = processor.download_image(file_name=file_name, url=content_url)
        resized = processor.resize_image(path)

        assert resized.shape == (224, 224, 3)

def test_load_vgg_model():
    import tensorflow as tf
    tf.config.experimental_run_functions_eagerly(True)
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    for layer in vgg.layers:
        print(layer.name)

def test_extract_vgg_layer():
    
    pass

def test_calc_content_loss():
    pass

def test_calc_style_loss():
    pass

def test_calc_total_loss():
    pass

def test_export_result():
    pass
