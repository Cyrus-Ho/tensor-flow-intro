import tensorflow as tf

# var0 = tf.Variable(0.0, dtype=tf.float32)
var0 = tf.Variable(0.0, dtype=tf.float32, name='var0')
print("var0 name:" + var0.name)

var0_again = tf.Variable(0.0, dtype=tf.float32, name='var0')
print("var0_again name:" + var0_again.name)

# var0 name:var0:0
# var0 name:var0_1:0
# in the same name scope use tf.Variable() and one variable name to create 2 variabls 
# in fact creates 2 different variables using 2 DIFFERENT variable names.


var_n_0_1 = tf.Variable(0.0, dtype=tf.float32, name='varn1')
with tf.name_scope('nscope1') as scope:
	var_n_1_1 = tf.Variable(0.0, dtype=tf.float32, name='varn1')
	var_n_1_1_again = tf.Variable(0.0, dtype=tf.float32, name='varn1')

print("var_n_0_1 name:" + var_n_0_1.name)
print("var_n_1_1 name:" + var_n_1_1.name)
print("var1_n_1_again name:" + var_n_1_1_again.name)

# using different name scopes to distinct 2 variables with the same names.
# in the same name scope use tf.Variable() and one variable name to create 2 variabls 
# in fact creates 2 different variables using 2 DIFFERENT variable names.

var_v_0_1 = tf.Variable(0.0, dtype=tf.float32, name='varv1')
print("var_v_0_1 name:" + var_v_0_1.name)
with tf.variable_scope('vscope1') as scope:
	var_v_1_1 = tf.Variable(0.0, dtype=tf.float32, name='varv1')
	print("var_v_1_1 name:" + var_v_1_1.name)
	var_v_1_1_again = tf.Variable(0.0, dtype=tf.float32, name='varv1')
	print("var_v_1_1_again name:" + var_v_1_1_again.name)

# using different name scopes to distinct 2 variables with the same names.
# in the same variable scope use tf.Variable() and one variable name to create 2 variabls 
# in fact creates 2 different variables using 2 DIFFERENT variable names.