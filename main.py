import tensorflow as tf
import imageio
import numpy as np
import os
import tempfile

import numpy as np
import tensorflow as tf

from styletf.data.data_utils import download_image, resize_image
from styletf.misc.sample_images import content_url, style_url
from styletf.model.network import StyleTF
from styletf.model.train import calc_content_loss, calc_style_loss, calc_total_loss, train_step
from styletf.utils import calc_gram_matrix


tf.config.experimental_run_functions_eagerly(True)

tf.random.set_seed(42)

style_layer = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layer = ['block5_conv2']

style_file_name = "./style.jpg"
style_path = download_image(file_name=style_file_name, url=style_url)
style_resized = resize_image(style_path)

content_file_name = "./content.jpg"
content_path = download_image(file_name=content_file_name, url=content_url)
content_resized = resize_image(content_path)

# input_image = tf.Variable(tf.random.normal(shape=style_resized.shape, mean=0.5, seed=42), dtype=tf.float32)
input_image = tf.Variable(content_resized, tf.float32)

style_tf = StyleTF(style_layer=style_layer, content_layer=content_layer)
optimizer = tf.keras.optimizers.Adam()

epochs = 1000
for epoch in range(epochs):
    loss = train_step(style_tf, optimizer, input_image, content_resized, style_resized)
    print("EPOCH: {} / LOSS: {}".format(epoch, loss))

    if epoch % 10 == 0:
        imageio.imwrite('out{}.jpg'.format(epoch), np.array(input_image.numpy()*255, np.uint8)[0, :, :, :])