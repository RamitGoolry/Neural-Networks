# Custom Activation Functions, Initializers, Regularizers and Constraints

### Activation Functions
```python
def my_softplus(z):
	return tf.math.log(tf.exp(z) + 1.0)
```

### Initializers
```python
def my_glorot_initializer(shape, dtype=tf.float32):
	stddev = tf.sqrt(2 / (shape[0] + shape[1]))
	return tf.random.normal(shape, stddev=stddev, dtype=dtype)
```

### Regularizers
```python
def my_l1_regularizer(weights): 
	return tf.reduce_sum(tf.abs(0.01 * weights))
```

### Constraints
```python
def my_positive_weights(weights):  # basically ReLU
	return tf.where(weights < 0, tf.zeros_like(weights), weights)
```

## Usage

```python
layer = keras.layers.Dense(
			30, activation=my_softplus,
			kernel_initializer = my_glorot_initializer,
			kernel_regularizer = my_l1_regularizer,
			kernel_constraint = my_positive_weights
	)
```