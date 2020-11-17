import tensorflow as tf
from bicubic_interp import bicubic_interp_2d
from keras import backend as K
import random 
def WarpST_one(x,inputs, name):
    out_size = (256,256,1)
    V = x
    U = inputs
#    U, V, out_size = x[0], x[1], x[2]
    """Deformable Transformer Layer with bicubic interpolation
    U : tf.float, [num_batch, height, width, num_channels].
        Input tensor to warp
    V : tf.float, [num_batch, height, width, 2]
        Warp map. It is interpolated to out_size.
    out_size: a tuple of two ints
        The size of the output of the network (height, width)
    ----------
    References :
      https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py
    """

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'+str(random.random())):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'+str(random.random())):
            # constants
            num_batch = tf.shape(im)[0]
#            height = tf.shape(im)[1]
#            width = tf.shape(im)[2]
            height = 256
            width = 256
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
#            x = (x - 0.5)* 4.0
#            y = (y - 0.5)* 4.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = tf.add(x0 ,1)
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = tf.add(y0, 1)

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = tf.multiply(width, height)
            base = _repeat(tf.range(tf.shape(im)[0])*dim1, out_height*out_width)
            base_y0 = tf.add(base , tf.multiply(y0,dim2))
            base_y1 = tf.add(base , tf.multiply(y1,dim2))
            idx_a = tf.add(base_y0 , x0)
            idx_b = tf.add(base_y1 , x0)
            idx_c = tf.add(base_y0 , x1)
            idx_d = tf.add(base_y1 , x1)

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims((tf.multiply((x1_f-x) , (y1_f-y))), 1)
            wb = tf.expand_dims((tf.multiply((x1_f-x) , (y-y0_f))), 1)
            wc = tf.expand_dims((tf.multiply((x-x0_f) , (y1_f-y))), 1)
            wd = tf.expand_dims((tf.multiply((x-x0_f) , (y-y0_f))), 1)
            output = tf.add_n([tf.multiply(wa,Ia), tf.multiply(wb,Ib), tf.multiply(wc,Ic), tf.multiply(wd,Id)])
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'+str(random.random())):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
#            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
#                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
#            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
#                            tf.ones(shape=tf.stack([1, width])))
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(0.0, 255.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(0.0, 255.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            grid = tf.concat([x_t_flat, y_t_flat], 0)
            return grid

    def _transform(V, U, out_size):
        with tf.variable_scope('_transform'+str(random.random())):
#            num_batch = tf.shape(U)[0]
            height = 256  #tf.shape(U)[1]
            width =  256 #tf.shape(U)[2]
#            num_channels = tf.shape(U)[3]

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
#            height_f = tf.cast(height, 'float32')
#            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)     # [2, h*w]
            grid = tf.reshape(grid, [-1])               # [2*h*w]
            grid = tf.tile(grid, tf.stack([tf.shape(U)[0]]))           # [n*2*h*w]
            grid = tf.reshape(grid, tf.stack([tf.shape(U)[0], 2, -1])) # [n, 2, h*w]

            # transform (x, y)^T -> (x+vx, x+vy)^T
#            V = bicubic_interp_2d(V, out_size)
            V = tf.transpose(V, [0, 3, 1, 2])           # [n, 2, h, w]
            V = tf.reshape(V, [tf.shape(U)[0], 2, -1])       # [n, 2, h*w]
            T_g = tf.add(V, grid)                       # [n, 2, h*w]

            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])
            input_transformed = _interpolate(
                U, x_s_flat, y_s_flat, out_size)
            output = tf.reshape(
                input_transformed, 
                tf.stack([tf.shape(U)[0], out_height, out_width, 1]))
            return output
#    name='DeformableTransformer'
    with tf.variable_scope(name):
        output = _transform(V, U, out_size)
        return output

def jitter_diff(jit):
    # second one reduce the first one become the difference of the jitter
    #input size is [batch, height, channel]
    jit1 = tf.slice(jit,[0,5,0],[-1,243,-1])
    jit2 = tf.slice(jit,[0,6,0],[-1,243,-1])
    dif = tf.subtract(jit1, jit2)
    return dif