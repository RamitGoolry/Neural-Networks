# Piecewise Constant Scheduling

Use a constant learning rate for a number of epochs (e.g. $\eta_0$ = 0.1 for 5 epochs) and then a smaller learning rate for another number of epochs (e.g. $\eta_1$ 0.005 for 50 epochs) and so on.

## Usage

We have to define a function and set it as a callback:

```python
def piecewise_constant():
	def piecewise_constant_fn(epoch):
		if epoch < 5:
			return 0.1
		elif epoch < 50:
			return 0.005
		else:
			return 0.001
	return piecewise_constant_fn

lr_scheduler = keras.callbacks.LearningRateScheduler(piecewise_constant())

history = model.fit(X_train, y_train, ..., callbacks = [lr_scheduler])
```