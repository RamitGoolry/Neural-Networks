# Monte Carlo Dropout

Monte Carlo Dropout is a method used on top of [[Dropout|Dropout Regularization]], to boost the accuracy of the readings.

The idea is to drop neurons out randomly during **testing**, and taking multiple predictions and then aggregate the results. 

The mean of all the output gives a more accurate prediction while the standard deviation gives us better uncertainty estimates.

## Usage
We replace the Keras `Dropout` layer by creating a sub-class `MCDropout`, which is then used in the model:

```python
class MCDropout(keras.layers.Dropout):
	def call(self, inputs):
		return super().call(inputs, training = True)
```

By forcing `training = True`, We are dropping out neurons during testing as well.

Now, in order to use Monte Carlo Dropout:
```python
y_probas = np.stack([model.predict(X_test) for sample in range(100)])
y_proba = y_probas.mean(axis = 0)
```

Essentially, we are making a 100 predictions, and then stack the predictions. Since dropout is active, all predictions will be different. 

Say the shape of our output is $10000 \times 10$, then our stacked output will have shape $100 \times 10000 \times 10$. Averaging over these will give us a better estimate than a single prediction.

#### Drawback
Since we are making a 100 predictions, Monte Carlo Dropout also takes a 100 times the time.