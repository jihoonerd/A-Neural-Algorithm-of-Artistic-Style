import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Lambda
from styletf.utils import calc_gram_matrix

class StyleTF(Model):
    # https://keras.io/applications/#extract-features-from-an-arbitrary-intermediate-layer-with-vgg19

    def __init__(self, style_layer, content_layer):
        super().__init__()
        self.vgg_model = tf.keras.applications.VGG19(include_top=False, pooling='avg', weights='imagenet')
        self.vgg_model.trainable = False
        self.style_layer = style_layer
        self.content_layer = content_layer
        self.style_model = Model(inputs=self.vgg_model.input, outputs=[self.vgg_model.get_layer(name).output for name in self.style_layer])
        self.content_model = Model(inputs=self.vgg_model.input, outputs=[self.vgg_model.get_layer(name).output for name in self.content_layer])

        self.scale255 = Lambda(lambda x: x * 255.0)

    def __call__(self, inputs):
        """
        input_shape: [1, 224, 224, 3] (white noise at the first time)
        """
        
        scaled = self.scale255(inputs)
        preprocessed = preprocess_input(scaled)

        style_outputs = self.style_model(preprocessed)
        style_gram_output = [calc_gram_matrix(style_output) for style_output in style_outputs]
        style_gram_dict = {layer:value for layer, value in zip(self.style_layer, style_gram_output)}

        content_outputs = self.content_model(preprocessed)
        content_dict = {layer:value for layer, value in zip(self.content_layer, content_outputs)}

        return {'style': style_gram_dict, 'content': content_dict}
