import os
import tempfile

import numpy as np
import tensorflow as tf

from styletf.data.data_utils import download_image, resize_image
from styletf.misc.sample_images import content_url, style_url
from styletf.model.network import StyleTFNetwork
from styletf.model.train import StyleTFTrain
from styletf.utils import calc_gram_matrix


content_file_name = "./content.jpg"
content_path = download_image(file_name=content_file_name, url=content_url)
content_resized = resize_image(content_path)

style_file_name = "./style.jpg"
style_path = download_image(file_name=style_file_name, url=style_url)
style_resized = resize_image(style_path)

content_layer = ['block5_conv2']
style_layer = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

trainer = StyleTFTrain(content_resized, style_resized, content_layer, style_layer)
trainer.train()