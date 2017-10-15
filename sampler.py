import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


# # Register the gradient for the mod operation. tf.mod() does not have a gradient implemented.
# @ops.RegisterGradient('histogram_fixed_width')
# def _mod_grad(op, grad):
#     x, y = op.inputs
#     gz = grad
#     x_grad = gz
#     y_grad = gz
#     return x_grad, y_grad
#
#
# tf.mod()


# empirical_hist = tf.gather(m, tf.nn.top_k(input=m, k=1).indices)
# true_hist = tf.gather(t_true_samples, tf.nn.top_k(input=t_true_samples, k=1).indices)

# value_range = tf.constant(value=(-4.0, 4.0), dtype=tf.float32)
# empirical_hist = tf.histogram_fixed_width(m, value_range, nbins=100)
# true_hist = tf.histogram_fixed_width(t_true_samples, value_range, nbins=100)
#
# empirical_hist = empirical_hist / np.sum(empirical_hist)
# true_hist = true_hist / np.sum(true_hist)

# loss = tf.reduce_mean(tf.square(m - t_true_samples))


def plot_density(x, it):
    plt.hist(x, bins=100)
    id = str(it).zfill(5)
    plt.suptitle("Iteration {}".format(id), size=16)
    plt.savefig('figs/{}.png'.format(id))  # save the figure to file
    plt.close()


NUM_SAMPLES = 4096

sess = tf.InteractiveSession()

t_seed = tf.placeholder(dtype=tf.float32, shape=(None, 1))
t_true_samples = tf.placeholder(dtype=tf.float32, shape=(None, NUM_SAMPLES))

m = slim.fully_connected(inputs=t_seed,
                         num_outputs=128,
                         activation_fn=tf.nn.tanh)
m = slim.fully_connected(inputs=m,
                         num_outputs=256,
                         activation_fn=tf.nn.tanh)
m = slim.fully_connected(inputs=m,
                         num_outputs=512,
                         activation_fn=tf.nn.tanh)
m = slim.fully_connected(inputs=m,
                         num_outputs=1024,
                         activation_fn=tf.nn.tanh)
m = slim.fully_connected(inputs=m,
                         num_outputs=2048,
                         activation_fn=None)
m = slim.fully_connected(inputs=m,
                         num_outputs=NUM_SAMPLES,
                         activation_fn=None)

empirical_hist = tf.nn.top_k(input=m, k=NUM_SAMPLES).values
true_hist = tf.nn.top_k(input=t_true_samples, k=NUM_SAMPLES).values

loss = tf.reduce_mean(tf.square(empirical_hist - true_hist))

train_step = tf.train.AdamOptimizer(0.001 * 0.05).minimize(loss)

sess.run(tf.global_variables_initializer())

for i in range(10000):
    bs = 100
    np_seed = np.random.rand(bs, 1)
    np_values = np.random.standard_exponential(size=(bs, NUM_SAMPLES))
    v_loss, _ = sess.run([loss, train_step], feed_dict={t_seed: np_seed,
                                                        t_true_samples: np_values})

    if i % 1 == 0:
        v_m = sess.run(m, feed_dict={t_seed: np.random.rand(1, 1)})[0]
        plot_density(v_m, i)
    print('i = {}, v_loss = {}'.format(i, v_loss))
