import os
import tempfile

import numpy as np
import tensorflow as tf

from styletf.data.data_utils import download_image, resize_image
from styletf.misc.sample_images import content_url, style_url
from styletf.model.network import StyleTFNetwork
from styletf.model.train import StyleTFTrain
from styletf.utils import calc_gram_matrix

tf.config.experimental_run_functions_eagerly(True)


def test_extract_vgg_layer():
    tf.random.set_seed(42)

    content_file_name = "test_content.jpg"
    content_path = download_image(file_name=content_file_name, url=content_url)
    content_resized = resize_image(content_path)

    style_file_name = "test_style.jpg"
    style_path = download_image(file_name=style_file_name, url=style_url)
    style_resized = resize_image(style_path)

    content_layer = ['block5_conv2']
    style_layer = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    trainer = StyleTFTrain(content_resized, style_resized, content_layer, style_layer, quick=False)
    content_out = trainer.model(content_resized)
    style_out = trainer.model(style_resized)

    assert 'content' in content_out
    assert 'style' in content_out
    assert 'content' in style_out
    assert 'style' in style_out

    assert len(content_out['content']) == 1
    assert len(content_out['style']) == 5
    assert len(style_out['content']) == 1
    assert len(style_out['style']) == 5

    assert content_out['content']['block5_conv2'].shape == (1, 14, 14, 512)
    assert content_out['style']['block5_conv1'].shape == (512, 512) # Gram matrix shape validity test
    assert style_out['content']['block5_conv2'].shape == (1, 14, 14, 512)
    assert style_out['style']['block5_conv1'].shape == (512, 512) # Gram matrix shape validity test

    os.remove("./source_images/test_content.jpg")
    os.remove("./source_images/test_style.jpg")

def test_calc_total_loss():
    tf.random.set_seed(42)

    content_file_name = "test_content.jpg"
    content_path = download_image(file_name=content_file_name, url=content_url)
    content_resized = resize_image(content_path)

    style_file_name = "test_style.jpg"
    style_path = download_image(file_name=style_file_name, url=style_url)
    style_resized = resize_image(style_path)

    content_layer = ['block5_conv2']
    style_layer = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    trainer = StyleTFTrain(content_resized, style_resized, content_layer, style_layer, quick=False)
    content_loss = trainer.calc_content_loss(trainer.model(trainer.content_image), trainer.model(trainer.input_image))
    style_loss = trainer.calc_style_loss(trainer.model(trainer.style_image), trainer.model(trainer.input_image))
    total_loss = trainer.calc_total_loss(content_loss, style_loss)

    assert content_loss == 14880.503
    assert style_loss == 25414748000000.0
    assert total_loss == 2.5414747e+16
    os.remove("./source_images/test_content.jpg")
    os.remove("./source_images/test_style.jpg")

def test_training():
    tf.random.set_seed(42)

    content_file_name = "test_content.jpg"
    content_path = download_image(file_name=content_file_name, url=content_url)
    content_resized = resize_image(content_path)

    style_file_name = "test_style.jpg"
    style_path = download_image(file_name=style_file_name, url=style_url)
    style_resized = resize_image(style_path)

    content_layer = ['block5_conv2']
    style_layer = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    trainer = StyleTFTrain(content_resized, style_resized, content_layer, style_layer, quick=True, train_epochs=5, log_interval=1)
    trainer.train()
    os.remove("./source_images/test_content.jpg")
    os.remove("./source_images/test_style.jpg")