# Custom Layers
## Layers that have no weights
We can use the `keras.layers.Lambda` to create any layer which is just a simple non trainable layer. Let us show this with a simple [[ReLU]] layer.

```python
relu_layer = keras.layers.Lambda(lambda x : max(0, x))
```

## Layers that have weights
We can subclass the `keras.layers.Layer` class to make a trainable layer.

```python
class MyDense(keras.layers.Layer):
	def __init__(self, units, activation=None, **kwargs):
		super().__init__(**kwargs)
		self.units = units
		self.activation = keras.activations.get(activation)
		
	def build(self, batch_input_shape):
		'''
		Used to build the Layer's weights, called the first time the layer 
		is used.
		'''
		self.kernel = self.add_weight(
						name = "kernel", 
						shape = [batch_input_shape[-1], self.units],
						initializer = "glorot_normal")
		self.bias = self.add_weight(
						name = "bias",
						shape = [self.units],
						initializer = "zeros")
		
		# Tells keras the layer is built, must be called at the end
		super.build(batch_input_shape) 
		
	def call(self, X):
		return self.actication(X @ self.kernel + self.bias)
	
	def compute_output_shape(self, batch_input_shape):
		'''
		Returns the shape of the layer's outputs
		'''
		return tf.TensorShape(batch_input_shape.as_list()[:-1] + self.units)
	
	def get_config(self):
		base_config = super().get_config()
		return {**base_config, "units" : self.units, 
				"activaiton" : keras.activations.serialize(self.activation)}
		
```

<font color = red>__NOTE :__</font> If your layer has a different behaviour during training and testing, like [[Batch Normalization]], then you must add a `train` argument in the `call()` function which will decide how the code operates. 