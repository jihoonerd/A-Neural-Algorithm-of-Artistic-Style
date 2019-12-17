import os

import imageio
import numpy as np
import tensorflow as tf

from styletf.model.network import StyleTFNetwork


class StyleTFTrain():

    def __init__(self, content_image, style_image, content_layer, style_layer):
        self.content_image = content_image
        self.style_image = style_image
        self.content_layer = content_layer
        self.style_layer = style_layer
        self.input_image = tf.Variable(tf.random.normal(shape=content_image.shape, mean=0.5, seed=42), dtype=tf.float32)
        self.model = StyleTFNetwork(content_layer=self.content_layer, style_layer=self.style_layer)
        self.optimizer = tf.keras.optimizers.Adam()

    
    def train(self, epochs: int=100):

        content_image = self.model(self.content_image)
        style_image = self.model(self.style_image)

        for epoch in range(epochs):
            loss = self.train_step(self.input_image, content_image, style_image)
            print("Epoch: {} / Loss: {}".format(epoch, loss))

            if epoch % 10 == 0:
                if not os.path.exists("./output/"):
                    os.mkdir("./output/")
                imageio.imwrite("./output/styletf_{}.jpg".format(epoch), np.array(self.input_image[0, :, :, :].numpy()*255, np.uint8))

    @tf.function
    def train_step(self, input_image, content_image, style_image):
        with tf.GradientTape() as tape:
            input_image = self.model(input_image)
            content_loss = self.calc_content_loss(content_image, input_image)
            style_loss = self.calc_style_loss(style_image, input_image)
            total_loss = self.calc_total_loss(content_loss, style_loss)

        grad = tape.gradient(total_loss, self.input_image)
        self.optimizer.apply_gradients([(grad, self.input_image)])
        self.input_image.assign(tf.clip_by_value(self.input_image, clip_value_min=0.0, clip_value_max=1.0))
        return total_loss
        
    @tf.function
    def calc_content_loss(self, target, input_img):
        """
        Paper formula (1)
        """
        content = target['content']
        input_content = input_img['content']

        content_loss = tf.add_n([tf.reduce_mean(tf.math.square(content[layer] - input_content[layer])) for layer in content.keys()])
        return content_loss

    @tf.function
    def calc_style_loss(self, target, input_img):
        """
        Paper formula (3), (4), (5)
        """
        style = target['style']
        input_style = input_img['style']

        style_loss = tf.add_n([tf.reduce_mean(tf.math.square(style[layer] - input_style[layer])) for layer in style.keys()])
        style_loss /= len(style)
        return style_loss

    @tf.function
    def calc_total_loss(self, content_loss, style_loss, alpha_to_beta=0.001):
        """
        Paper formual (7)
        """
        total_loss = content_loss + (1/alpha_to_beta) * style_loss
        return total_loss
