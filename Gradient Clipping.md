# Gradient Clipping

Another method to mitigate the [[Vanishing and Exploding Gradients problem]] (Exploding Gradients only) is to *clip* the gradients.

This means that the values of the components of the gradients are limited to a certain upper/lower bound, beyond which the values can not exceed.

Therefore, if we have a vector like $\begin{bmatrix} 0.99 \\ 100 \end{bmatrix}$ and `clipvalue = 1.0`, the vector will settle down to $\begin{bmatrix} 0.99 \\ 1.0 \end{bmatrix}$.

Gradient Clipping is genreally used in [[Recurrent Neural Networks]], where [[Batch Normalization]] is not easy to use.

## Usage

```python
optimizer = keras.optimizers.SGD(clipvalue=1.0)
model.compile(loss='mse', optimizer=optimizer)
```