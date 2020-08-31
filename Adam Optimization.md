# Adam Optimization

Adam stands for __adaptive moment estimation__.

It combines the ideas of [[RMSProp]] and [[Momentum Optimization]]:
1. It keeps track of exponenitally decaying gradients like [[Momentum Optimization]].
2. It keeps track of the squares of exponentially decaying gradients like [[RMSProp]].


## The Algorithm

1. $$m \leftarrow \beta_1m - (1-\beta_1) \nabla_\theta J(\theta)$$
2. $$s \leftarrow \beta_2s - (1 - \beta_2) \nabla_\theta J(\theta) \otimes \nabla_\theta J(\theta)$$
3. $$\hat m \leftarrow \frac{m}{1 - \beta_1^t}$$
4. $$\hat s \leftarrow \frac{s}{1 - \beta_2^t}$$
5. $$\theta \leftarrow \theta + \eta \hat m \oslash \sqrt{\hat s + \varepsilon}$$

Step 1 is very similar to [[Momentum Optimization]] while step 2 is reminiscent of [[RMSProp]].

The momentum decay hyperparamter $\beta_1$ is typically initialised to $0.9$, while the scale decay hyperparameter $\beta_2$ is set to $0.999$.

## Usage
```python
optimizer = keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
```
[TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)
### Variants
- [[AdaMax]]
- [[Nadam]]