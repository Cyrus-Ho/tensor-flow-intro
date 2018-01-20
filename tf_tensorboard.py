import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# conflict between tensorflow 1.0 CUDA 8.0
# export LD_LIBRARY_PATH=$/usr/local/cuda/lib64:$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64


def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

def variable_summaries(var):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			weights = weight_variable([input_dim, output_dim])
			print('weight:', weights.name)
			variable_summaries(weights)
		with tf.name_scope('biases'):
			biases = weight_variable([output_dim])
			print('biases:', biases.name)
			variable_summaries(biases)
		with tf.name_scope('Wx_plus_b'):
			preactivate = tf.matmul(input_tensor, weights) + biases
			print('Wx+b:', preactivate.name)
			tf.summary.histogram('pre_activiation', preactivate)
		activations = act(preactivate, name='activation')
		print('activation:', activations.name)
		tf.summary.histogram('activations', activations)
		return activations

def feed_dict(train):
	if train:
		xs, ys = mnist.train.next_batch(100)
		k = dropout
	else:
		xs, ys = mnist.test.images, mnist.test.labels
		k = 1.0
	return {x: xs, y_: ys, keep_prob: k}



print('start...')
max_steps = 10000
learning_rate=0.001
dropout=0.9
# data_dir='/home/pi/practice/tf_practice/mnist/input_data'
log_dir='/tmp/tensorflow/mnist/logs/mnist_with_summaries'

print('start download mnist')
mnist = input_data.read_data_sets('./mnist',one_hot=True)
print('finish download mnist')
sess = tf.InteractiveSession()

with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, [None, 784], name='x-input')
	y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
	print('x:', x.name)
	print('y_:', y_.name)

with tf.name_scope('input_reshape'):
	image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
	tf.summary.image('input', image_shaped_input, 10)

hidden1 = nn_layer(x, 784, 500, 'layer1')
print('hidden1:', hidden1.name)

with tf.name_scope('dropout'):
	keep_prob = tf.placeholder(tf.float32)
	print('keep_p:', keep_prob.name)
	tf.summary.scalar('dropout_keep_probability', keep_prob)
	dropped = tf.nn.dropout(hidden1, keep_prob)
	print('dropped:', dropped.name)

y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)
print('y:', y.name)

with tf.name_scope('cross_entropy'):
	diff = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
	print('diff:', diff.name)
	with tf.name_scope('totol'):
		cross_entropy = tf.reduce_mean(diff)
	print('cross_entropy:', cross_entropy.name)

tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
	print('train_step:', train_step.name)
with tf.name_scope('accuracy'):
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		print('correct prediction:', correct_prediction.name)
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print('accuracy', accuracy.name)

tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
print('merged:', merged)
train_writter = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writter = tf.summary.FileWriter(log_dir + '/test')
tf.global_variables_initializer().run()



saver = tf.train.Saver()
for i in range(max_steps):
	if 0 == i % 10:
		summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
		test_writter.add_summary(summary, i)
		print('Accuracy at step %s: %s' % (i, acc))
	else:
		if i % 100 == 99:			
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
			summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True), options=run_options, run_metadata=run_metadata)
			train_writter.add_run_metadata(run_metadata, 'step%03d' % i)
			train_writter.add_summary(summary, i)
			saver.save(sess, log_dir+"/model.ckpt", i)
			print('Adding run metadata for', i)
		else:
			summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
			train_writter.add_summary(summary, i)
train_writter.close()
test_writter.close()


