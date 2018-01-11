import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet


print('Loading traffic signs data.')
training_file = 'train.p'

with open(training_file, mode='rb') as f:
  train = pickle.load(f)


print('Splitting data into training and validation sets.')
X_train, y_train = train['features'], train['labels']

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
						      test_size=0.2)


print('Defining placeholders and resize operation.')
nb_classes = 43
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, nb_classes)


print('Passing placeholder as first argument to `AlexNet`.')
fc7 = AlexNet(resized, feature_extract=True)


# `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)


print('Adding the final layer for traffic sign classification.')
shape = (fc7.get_shape().as_list()[-1], nb_classes)

fc8W = tf.Variable(tf.truncated_normal(shape, 0, 0.1))
fc8b = tf.Variable(tf.zeros(shape[1]))

fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


print('Defining loss, training, accuracy operations.')
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y,
							logits=fc8)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])


correct_prediction = tf.equal(tf.argmax(fc8, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print('Training and evaluating the feature extraction model.')
with tf.Session() as sess:
  BATCH_SIZE = 128
  num_examples = len(X_train)

  sess.run(tf.global_variables_initializer())
  
  for offset in range(0, num_examples, BATCH_SIZE):
    print('Batch', int(offset / BATCH_SIZE),
          'of', int(num_examples / BATCH_SIZE))
    end = offset + BATCH_SIZE
    batch_X, batch_y = X_train[offset:end], y_train[offset:end]
    sess.run(training_operation, feed_dict={x: batch_X, y: batch_y})
    break 
  
  total_accuracy = 0
  for offset in range(0, num_examples, BATCH_SIZE):
    end = offset + BATCH_SIZE
    batch_X, batch_y = X_valid[offset:end], y_valid[offset:end]
    accuracy = sess.run(accuracy_operation, feed_dict={x: batch_X, y: batch_y})
    total_accuracy += (accuracy * len(batch_X))
    break

  total_accuracy = total_accuracy / num_examples

print('Validation accuracy = {:.3f}'.format(accuracy))

