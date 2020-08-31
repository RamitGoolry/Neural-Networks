# Max-Norm Regularization

Max-Norm Regularization is another technique for [[Regularization]]. 

For each neuron, it constrains the weights $w$ of the incoming connections such that $$\| w \|_2 \le r$$ where $r$ is a limiting max-norm hyperparameter and $\| \cdot \|_2$ is the $\ell_2$ norm.

So, in order to regularize a $w$ greater than $r$,
$$w \leftarrow w \cdot \frac{r}{\| w \|_2}$$

The smaller the value of $r$, the more the regularization, and as a result it reduces overfitting.

## Usage
We simply need to set the `kernel_constraint` parameter of a layer with the `max_norm()` function as: 

```python
keras.layers.Dense(
	100, activation='elu', kernel_initializer='he_normal',
	kernel_constraint=keras.constraints.max_norm(1.)
)
```

[TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/constraints/MaxNorm)