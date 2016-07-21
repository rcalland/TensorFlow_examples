import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# disable GPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# parameters
epochs = 1000
true_mu = 1.0
true_sigma = 0.5
n_samples = 10
seed_mu = 0.0
seed_sigma = 1.0

# plot the results
def plot_samples(s, mu, sigma):
    count, bins, ignored = plt.hist(s, 30, normed=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
             np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
             linewidth=2, color='r')
    plt.show()

# evaluate a gaussian density
def gauss(X, mu, sigma):
    return (1.0 / (sigma * tf.sqrt(2 * np.pi)) * tf.exp(-tf.square(X - mu) / (2 * tf.square(sigma))) )

# generate some data
samples = np.random.normal(true_mu, true_sigma, n_samples)

# define variables for tensorflow
mu = tf.Variable(seed_mu, name="mean")
sigma = tf.Variable(seed_sigma, name="sigma")

# a placeholder that we will feed our data samples into
X = tf.placeholder("float")

# negative log likelihood 
nglogl = -tf.log( gauss(X, mu, sigma) ) 

# learning procedure, i.e. minimize the -nglogl 
learn = tf.train.AdamOptimizer(0.01).minimize(tf.reduce_sum(nglogl))

# start up the session
sess = tf.Session()

# need to init the variables, otherwise TF will cry
tf.initialize_all_variables().run(session=sess)

# now interate many times over our data, and find the best parameters
for i in range(epochs):
    result = sess.run(learn, feed_dict={X: samples})

# print the best fit results
print(sess.run(mu), sess.run(sigma))

# plot the best fit over the samples
plot_samples(samples, sess.run(mu), sess.run(sigma))

# RELEASE THE SESSION
sess.close()
