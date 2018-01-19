import tensorflow as tf

var0 = tf.Variable(0.0, dtype=tf.float32, name='var0')
print("var0 name:" + var0.name)
var0_again = tf.get_variable(shape=[1], name='var0')
print("var0_again name:" + var0_again.name)
#  2 different variables

var1 = tf.get_variable(shape=[1], name='var1')
print("var1 name:" + var1.name)
try:
	var1_again = tf.get_variable(name='var1')
	print("var1_again name:" + var1_again.name)
except:
	print("ERROR!")

with tf.variable_scope('vscope_no_reuse') as scope:
	print("scope name:" + scope.name)
	print("scope reuse:", scope.reuse)
	var2 = tf.get_variable(shape=[1], name='var2')
	print("var2 name:" + var2.name)
	try:
		var2_again = tf.get_variable(name='var2')
		print("var2_again name:" + var2_again.name)
	except:
		print("ERROR!")

with tf.variable_scope('vscope_reuse') as scope:
	print("scope name:" + scope.name)
	print("scope reuse:", scope.reuse)

	var3 = tf.get_variable(shape=[1], name='var3')
	print("var3 name:" + var3.name)
	
	scope.reuse_variables()
	print("scope reuse after set True:", scope.reuse)
	
	try:
		var3_again = tf.get_variable(name='var3')
		print("var3_again name:" + var3_again.name)
	except:
		print("ERROR!")