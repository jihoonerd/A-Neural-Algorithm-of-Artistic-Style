import tensorflow as tf


def calc_content_loss(content, input):
    """
    Paper formula (1)
    https://www.tensorflow.org/api_docs/python/tf/math/add_n
    """
    content_loss = tf.add_n([tf.reduce_mean(tf.math.square(content[layer] - input[layer])) for layer in content.keys()])
    return content_loss

def calc_style_loss(style, input):
    """
    Paper formula (3), (4), (5)
    """
    style_loss = tf.add_n([tf.reduce_mean(tf.math.square(style[layer] - input[layer])) for layer in style.keys()])
    style_loss /= len(style)
    return style_loss

def calc_total_loss(content_loss, style_loss, alpha_to_beta=0.001):

    total_loss = content_loss + style_loss

    return total_loss