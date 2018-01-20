import math
from datetime import datetime
import time
import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim

def time_tensorflow_run(session, target, info_string):
	num_steps_burn_in = 10
	total_duration = 0.0
	total_duration_squared = 0.0

	for i in range(num_batches + num_steps_burn_in):
		start_time = time.time()
		_ = session.run(target)
		duration = time.time() - start_time
		if i >= num_steps_burn_in:
			if not i % 10:
				print('%s: step %d, duration = %.3f' %(datetime.now(), i - num_steps_burn_in, duration))
			total_duration += duration
			total_duration_squared += duration * duration

	mn = total_duration / num_batches
	vr = total_duration_squared / num_batches - mn * mn
	sd = math.sqrt(vr)
	print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, num_batches, mn, sd))

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'arg'])):
	'A named tuple describing a ResNet block'

def subsample(inputs, factor, scope=None):
	if 1 == factor:
		return inputs
	else:
		return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
	if 1 == stride:
		return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', scope=scope)
	else:
		pad_total = kernel_size - 1
		pad_beg = pad_total // 2
		pad_end = pad_total - pad_beg
		inputs_pad = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
		return slim.conv2d(inputs_pad, num_outputs, kernel_size, stride=stride, padding='VALID', scope=scope)

@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None):
	for idx, block in enumerate(blocks):
		# print('block %d:' % (idx+1))
		with tf.variable_scope(block.scope, 'block', [net]) as sc:
			print(sc.name)
			for i, unit in enumerate(block.arg):
				with tf.variable_scope('unit_%d' % (i+1), values=[net]) as sc_unit:
					print(sc_unit.name+': '+str(unit))
					unit_depth, unit_depth_bottleneck, unit_stride = unit
					net = block.unit_fn(net, depth=unit_depth, 
						depth_bottleneck=unit_depth_bottleneck, 
						stride=unit_stride)
			net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
		print(net.get_shape())
		print()
	return net

# define default arg scope
def resnet_arg_scope(is_training=True,
	weight_decay=0.0001,
	batch_norm_decay=0.997,
	batch_norm_epsilon=1e-5,
	batch_norm_scale=True):
	
	batch_norm_params = {
		'is_training': is_training,
		'decay': batch_norm_decay,
		'epsilon': batch_norm_epsilon,
		'scale': batch_norm_scale,
		'updates_collections': tf.GraphKeys.UPDATE_OPS
	}

	with slim.arg_scope([slim.conv2d],
		weights_regularizer=slim.l2_regularizer(weight_decay),
		weights_initializer=slim.variance_scaling_initializer(),
		activation_fn=tf.nn.relu,
		normalizer_fn=slim.batch_norm,
		normalizer_params=batch_norm_params):
		with slim.arg_scope([slim.batch_norm], **batch_norm_params):
			with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
				return arg_sc

# define a residual block
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections=None, scope=None):
	with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
		depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
		preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

		# deal with the shortcut
		if depth == depth_in:
			shortcut = subsample(inputs, stride, 'shortcut')
		else:
			shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None, scope='shortcut')

		# deal with the residual path
		residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
		residual = conv2d_same(residual, depth_bottleneck, 3, stride, scope='conv2')
		residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')

		output = shortcut + residual
		return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)

# define a general resnet given blocks info
def resnet_v2(inputs, blocks, num_classes=None, global_pool=True, include_root_block=True, reuse=None, scope=None):
	with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
		# 'print(sc.name):', print(sc.name)
		end_points_collection = sc.original_name_scope + '_end_points'
		print('end_points_collection:', end_points_collection)
		with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense], outputs_collections=end_points_collection):
			net = inputs
			if include_root_block:
				with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
					net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
				net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
			net = stack_blocks_dense(net, blocks)
			net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
			if global_pool:
				net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
			if num_classes is not None:
				net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
			end_points = slim.utils.convert_collection_to_dict(end_points_collection)
			if num_classes is not None:
				end_points['predictions'] = slim.softmax(net, scope='predictions')
			return net, end_points

def resnet_v2_50(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_50'):
	blocks = [
		Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
		Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
		Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
		Block('block4', bottleneck, [(2048, 512, 1)] * 3)
	]
	return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)

def resnet_v2_152(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_152'):
	blocks = [
		Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
		Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
		Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
		Block('block4', bottleneck, [(2048, 512, 1)] * 3)
	]
	return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)


batch_size = 32
height, width = 224, 224
inputs = tf.random_uniform((batch_size, height, width, 3))

# a = conv2d_same(inputs, 10, 3, 4, 'scope_test')
# print(a.name)
# print(a.shape)

with slim.arg_scope(resnet_arg_scope(is_training=False)):
	net, end_points = resnet_v2_152(inputs, 1000)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches = 100
time_tensorflow_run(sess, net, "Forward")

