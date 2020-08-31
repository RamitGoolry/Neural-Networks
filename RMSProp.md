# RMSProp

RMSProp solves the problem faced by [[AdaGrad]] that it stops before reaching the global optimum. It does this by accumulating only the gradients of the most recent iterations, as compared to [[AdaGrad]], which considers all of the gradients.

It does so by using exponential decay.

## Usage
Keras has an `RMSProp` optimizer
```python
optimizer = keras.optimizers.RMSProp(lr=0.001, rho=0.9)
```

This optimizer almost always performs better than [[AdaGrad]].