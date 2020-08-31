# Dropout

Dropout is one of the most popular regularization techniques, and performs very well in most networks.

The idea is that at each training step, each neuron has the probability to be "dropped out" or turned off of training temporarily, with a probability $p$, called the _dropout rate_. 

### Why does this work?
Neurons trained with dropout have to be more robust, since they can not rely on any other neurons during training to do the heavy lifting. They must be as useful as they can on their own.

### Clarification
There is one important caveat here : 

Suppose $p = 50\%$, in which case during testing, the inputs are double the size than in training. To compensate for this, we must half each neurons input weights so that the signal is similar to what it was. Otherwise, the network might not perform well.

More generally, we must multiple each input connection by the _keep probability_ $(1- p)$ after training.

## Usage

To implement dropout, we can use the `keras.layers.Dropout` layer as:
```python
model = keras.models.Sequential([
	keras.layers.Flatten(input_shape=[28, 28]),
	keras.layers.Dropout(rate=0.2),
	keras.layers.Dense(300, activation="elu"),
	keras.layers.Dropout(rate=0.2),
	keras.layers.Dense(100, activation="elu"),
	keras.layers.Dropout(rate=0.2),
	keras.layers.Dense(10, activation="softmax")
])
```
[TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout)

In general, if the model is overfitting, we can increase dropout, and if the model is underfitting we can reduce dropout.

The effect of Dropout can be further amplified by using [[Monte Carlo Dropout]].
