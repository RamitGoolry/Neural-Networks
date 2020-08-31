# Saving and Loading Custom Components
`keras.models.load_model` supports custom objects, which need to be provided with a dictionary mapping:

```python
model = keras.models.load_model(
			"my_model_with_a_custom_loss.h5",
			custom_objects = {"huber_fn" : huber_fn}
		)
```

However, such models can not have any hyperparameters associated with them, since we can not pass them in. This can be fixed by returning functions:

```python
def create_huber(threshold = 1.0):
	def huber_fn(y_true, y_pred):
		error = y_true - y_pred
		is_small_error = tf.abs(error) < 1
		squared_error = tf.square(error) / 2
		linear_loss = tf.abs(error) - 0.5
	
		return tf.where(is_small_error, squared_loss, linear_loss)
	return huber_fn
	
model.compile(loss = create_huber(2.0))
```

This still has a problem. The hyperparameters are not saved in the model. This is fixed by creating a subclass of `keras.losses.Loss` class, and implementing its `get_config()` method:

```python
class HuberLoss(keras.losses.Loss):
	def __init__(self, threshold=1.0, **kwargs):
		self.threshold = threshold
		super().__init__(**kwargs)
	
	def call(self, y_true, y_pred):
		error = y_true - y_pred
		is_small_error = tf.abs(error) < 1
		squared_error = tf.square(error) / 2
		linear_loss = tf.abs(error) - 0.5
	
		return tf.where(is_small_error, squared_loss, linear_loss)
	
	def get_config(self):
		base_config = super().get_config()
		return {**base_config, "threshold" : self.threshold}
```

Now, the class can be used however you want. 