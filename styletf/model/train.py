import tensorflow as tf
from styletf.model.network import StyleTFNetwork

class StyleTFTrain():

    def __init__(self, content_image, style_image, content_layer, image_layer):
        self.content_image = content_image
        self.style_image = style_image
        self.content_layer = content_layer
        self.style_layer = style_layer
        self.input_image = tf.Variable(tf.random.normal(shape=style_resized.shape, mean=0.5, seed=42), dtype=tf.float32)
        self.model = StyleTFNetwork(content_layer=self.content_layer, style_layer=self.style_layer)
        self.optimizer = tf.keras.optimizers.Adam()

    def train(self, input_image, epochs: int=100):


        for epoch in range(epochs):

            loss = self.train_step(self.input_image, self.conten)

    @tf.function
    def train_step(self, input_img, content_img, style_img):
        with tf.GradientTape() as tape:
            content_loss = self.calc_content_loss(self.model(self.content_image), self.model(self.input_image))
            style_loss = self.calc_style_loss(self.model(self.style_image), self.model(self.input_image))
            total_loss = self.calc_total_loss(content_loss, style_loss)
        
    @tf.function
    def calc_content_loss(target, input_img):
        """
        Paper formula (1)
        https://www.tensorflow.org/api_docs/python/tf/math/add_n
        """
        content = target['content']
        input_content = input_img['content']

        content_loss = tf.add_n([tf.reduce_mean(tf.math.square(content[layer] - input_content[layer])) for layer in content.keys()])
        return content_loss

    @tf.function
    def calc_style_loss(target, input_img):
        """
        Paper formula (3), (4), (5)
        """
        style = target['style']
        input_style = input_img['style']

        style_loss = tf.add_n([tf.reduce_mean(tf.math.square(style[layer] - input_style[layer])) for layer in style.keys()])
        style_loss /= len(style)
        return style_loss

    @tf.function
    def calc_total_loss(content_loss, style_loss, alpha_to_beta=0.001):
        """
        Paper formual (7)
        """
        total_loss = content_loss + (1/alpha_to_beta) * style_loss
        return total_loss


@tf.function
def calc_content_loss(target, input_img):
    """
    Paper formula (1)
    https://www.tensorflow.org/api_docs/python/tf/math/add_n
    """
    content = target['content']
    input_content = input_img['content']

    content_loss = tf.add_n([tf.reduce_mean(tf.math.square(content[layer] - input_content[layer])) for layer in content.keys()])
    return content_loss

@tf.function
def calc_style_loss(target, input_img):
    """
    Paper formula (3), (4), (5)
    """
    style = target['style']
    input_style = input_img['style']

    style_loss = tf.add_n([tf.reduce_mean(tf.math.square(style[layer] - input_style[layer])) for layer in style.keys()])
    style_loss /= len(style)
    return style_loss

@tf.function
def calc_total_loss(content_loss, style_loss, alpha_to_beta=0.001):
    """
    Paper formual (7)
    """
    total_loss = content_loss + (1/alpha_to_beta) * style_loss
    return total_loss

@tf.function
def train_step(model, optimizer, input_img, content_img, style_img):

    with tf.GradientTape() as tape:
        tape.watch(input_img)
        tape.watch(style_img)
        tape.watch(content_img)
        style_target = model(style_img)
        content_target = model(content_img)
        output = model(input_img)

        style_loss = calc_style_loss(style_target, output)
        content_loss = calc_content_loss(content_target, output)
        loss = calc_total_loss(content_loss, style_loss)
    
    grad = tape.gradient(loss, input_img)
    optimizer.apply_gradients([(grad, input_img)])
    input_img.assign(tf.clip_by_value(input_img, clip_value_min=0.0, clip_value_max=1.0))
    return loss