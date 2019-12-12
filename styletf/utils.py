import tensorflow as tf

def calc_gram_matrix(input_mat):
    
        

    matrix_t = tf.transpose(input_mat)
    gram_matrix = tf.linalg.matmul(matrix_t, input_mat)
    return gram_matrix