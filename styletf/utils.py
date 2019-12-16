import tensorflow as tf

@tf.function
def calc_gram_matrix(input_mat):
    """
    Paper directly mentions about calculating Gram matrix:

    G_{ij}^l = \sum_k F_{ij}^l F_{jk}^l

    i and j stand for filter position and k stands for position in each filters.
    
    If matrix A is composed of vectors, a1, a2, a3, etc,
    e.g. A = [a1, a2, a3, ...] note that a1, a2, a3 are column vecdtors
    then Gram matrix G can be calculated as $G = A^T cdot A$

    inputs:
        It takes input shape of [1, height, width, channel]
    returns:
        [1, channel, channel, 1]
    """
    
    channel_size = input_mat.shape[-1]

    # From [1, height, width, channel] to [1, height * width, channel]
    vectorized_input = tf.reshape(input_mat, [1, -1, channel_size])
    # Transform it to shape of [channel, height * width]
    mat_2d = vectorized_input[0, :, :]
    F = tf.transpose(mat_2d)
    # Calculate gram matrix
    gram_mat = tf.linalg.matmul(F, mat_2d) # this produce the shape of [channel, channel]
    denominator = input_mat.shape[1] * input_mat.shape[2]
    return gram_mat / denominator