# Custom Metrics

Defining a custom metric is much like defining [[Custom Loss Functions]].

Let us take an example of [[Precision]], to demonstrate how we can write a custom metric.

```python
>>> precision = keras.metrics.Precision()
>>> precision([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1])
	<tf.Tensor: id=581729, shape=(), dtype=float32, numpy=0.8>
>>> precision([0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0])
	<tf.Tensor: id=581780, shape=(), dtype=float32, numpy=0.5>
```

In the example above, we see 2 values, 0.8 and 0.5, while the individual precision values of each prediction are 80% and 0% respectively. We arrive to the 0.5 through compounding precision values.

###### Q: Shouldn't the compunded value be 0.4?
No, the 0.5 calculation is made as $\frac{80 + 0}{100 + 100}$, as compared to 0.4, which would be $\frac{\frac{80}{100} + \frac{0}{100}}{2}$. 

This kind of metric where the aggregation doesn't work by simply averaging is called a _streaming metric_.

---
To create a streaming metric we can subclass the `keras.Metric` class. 
Let's demonstrate this with the [[Huber Loss|Huber Metric]].

```python

class HuberMetric(keras.metrics.Metric):
	def __init__(self, threshold = 1.0, **kwargs):
		super().__init__(**kwargs) # Handle everything to do with Keras Metric instantiation
		self.huber_fn = create_huber(threshold)
		self.total = self.add_weight("total", initializer = "zeros")
		self.count = self.add_weight("count", initializer = "zeros")

	def update_state(self, y_true, y_pred, sample_weight = None):
		metric = self.huber_fn(y_true, y_pred)
		self.total.assign_add(tf.reduce_sum(metric))
		self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
	
	def result(self):
		return self.total / self.count
	
	def get_config(self):
		base_config = super().get_config()
		return {**base_config, "threshold" : self.threshold}
```


__Further:__
[[Difference between Metrics and Losses]]