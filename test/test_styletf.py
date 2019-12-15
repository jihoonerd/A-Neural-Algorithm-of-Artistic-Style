import os
import tempfile
import numpy as np
import tensorflow as tf

from styletf.data.data_utils import download_image, resize_image
from styletf.misc.sample_images import content_url, style_url
from styletf.model.network import StyleTF
from styletf.utils import calc_gram_matrix

tf.config.experimental_run_functions_eagerly(True)


def test_resize_img():

    with tempfile.TemporaryDirectory() as tmpdir:
        file_name = tmpdir + "/content.jpg"
        path = download_image(file_name=file_name, url=content_url)
        resized = resize_image(path)

        assert resized.shape == (1, 224, 224, 3)

def test_load_vgg_model():
    styletf = StyleTF(style_layer=['block1_conv1'], content_layer=['block5_conv1'])
    layer_names = [layer.name for layer in styletf.vgg_model.layers]
    layer_list = ['input_1',
                  'block1_conv1', 'block1_conv2', 'block1_pool',
                  'block2_conv1', 'block2_conv2', 'block2_pool',
                  'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4', 'block3_pool',
                  'block4_conv1','block4_conv2',  'block4_conv3', 'block4_conv4', 'block4_pool',
                  'block5_conv1','block5_conv2','block5_conv3','block5_conv4', 'block5_pool',
                  'global_average_pooling2d']
    assert layer_list == layer_names
    assert len(layer_names) == 23
    assert layer_names[-1] == "global_average_pooling2d"
    assert layer_names[0] == "input_1"

def test_extract_vgg_layer():
    style_layer = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = ['block5_conv1']
    with tempfile.TemporaryDirectory() as tmpdir:
        file_name = tmpdir + "/content.jpg"
        path = download_image(file_name=file_name, url=style_url)
        resized = resize_image(path)

        style_tf = StyleTF(style_layer=style_layer, content_layer=content_layer)
        outputs = style_tf(resized)

        assert 'content' in outputs
        assert 'style' in outputs
        assert len(outputs['content']) == 1
        assert len(outputs['style']) == 5
        assert outputs['style']['block5_conv1'].shape == (512, 512) # Gram matrix shape validity test
        assert outputs['content']['block5_conv1'].shape == (14, 14, 512)
def test_calc_content_loss():
    pass

def test_calc_style_loss():
    pass

def test_calc_total_loss():
    pass

def test_export_result():
    pass
