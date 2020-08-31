# Custom Loss Functions

You can write your own custom loss functions in TensorFlow, if they are not available in TensorFlow to begin with, or if you want to tweak a preexisting loss function. 

## Usage

Let's use the example of [[Huber Loss]] to write a custom loss function.

```python
def huber_fn(y_true, y_pred):
	error = y_true - y_pred
	is_small_error = tf.abs(error) < 1
	squared_error = tf.square(error) / 2
	linear_loss = tf.abs(error) - 0.5
	
	return tf.where(is_small_error, squared_loss, linear_loss)
```

Now, this function can be used to compile the Keras model as :

```python
model.compile(loss = huber_fn, optimizer = 'nadam')
model.fit(X_train, y_train, [...])
```