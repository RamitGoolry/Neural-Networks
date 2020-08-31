# Custom Models
Let us say we want to make a `ResidualRegressor`, i.e., a Regressor consisting of residual block : 

![[Residual Regressor.png]]

We need to first construct a `ResidualBlock` which consists of some Dense Layers and a matrix addition operation, which is a [[Custom Layers|Custom Layer]]:
```python

class ResidualBlock(keras.layer.Layer):
	def __init__(self, n_layers, n_neurons, **kwargs):
		super().__init__(**kwargs)
		self.hidden = [keras.layers.Dense(n_neurons, activation = 'elu',
										  kernel_initializer = 'he_normal')
						for _ in range(n_layers)]
		
	def call(self, inputs):
		Z = inputs # deep copy, its probably a good idea to use copy.deepcopy()
		for layer in self.hidden:
			Z = layer(Z)
		return inputs + Z
```

Now the model can be constructed as:

```python
class ResidualRegressor(keras.Model):
	def __init__(self, output_dim, **kwargs):
		self().__init__(**kwargs)
		self.hidden1 = keras.layers.Dense(30, activation = 'elu', kernel_initializer = 'he_normal')
		
		self.block1 = ResidualBlock(2, 30)
		self.block2 = ResidualBlock(2, 30)
		
		self.out = keras.layers.Dense(output_dim)
	
	def call(self, inputs):
		Z = self.hidden1(inputs)
		
		for _ in range(3): # Goes through block1 thrice
			Z = self.block1(Z)
			
		Z = self.block2(Z)
		
		return self.out(Z)
```