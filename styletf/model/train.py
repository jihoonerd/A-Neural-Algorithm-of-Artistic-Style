import tensorflow as tf


class StyleTFTrain():

    def __init__(self, content_image, style_image, content_layer, image_layer):
        pass

    def calc_content_loss(self, content, input_img)

    def calc_style_loss(self, style, input_img)

    def calc_total_loss(self, input)

    def train_step(self)



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