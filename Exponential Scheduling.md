# Exponential Scheduling

We set the learning rate to a funciton of the iteration number $t$:

$$\eta(t) = \eta_0 \ 0.1^{\frac{t}{s}}$$

Where:
- $\eta_0$ is the initial learning rate
- $s$ is the number of steps

After every $s$ steps, the learning rate drops from $\frac{\eta_0}{1}$ to $\frac{\eta_0}{10}$ to $\frac{\eta_0}{100}$ and so on.

## Usage

We have to define a function and set it as a callback:

```python
def exponential_decay(lr0, s):
	def exponential_decay_fn(epoch):
		return lr0 * 0.1**(epoch/s)
	return exponential_decay_fn

lr_scheduler = keras.callbacks.LearningRateScheduler(
	exponential_decay(lr0 = 0.01, s = 20)
)

history = model.fit(X_train, y_train, ..., callbacks = [lr_scheduler])
```