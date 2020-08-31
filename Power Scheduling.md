# Power Scheduling

We set the learning rate to a funciton of the iteration number $t$:

$$\eta(t) = \frac{\eta_0}{(1 + \frac{t}{s})^c}$$

Where:
- $\eta_0$ is the initial learning rate
- $c$ is a constant power
- $s$ is the number of steps

After every $s$ steps, the learning rate drops from $\frac{\eta_0}{1}$ to $\frac{\eta_0}{2}$ to $\frac{\eta_0}{3}$ and so on.

## Usage
We set the `decay` parameter when creating an optimizer

```python
optimizer = keras.optimizers.SGD(lr = 0.01, decay = 1e-4)
```