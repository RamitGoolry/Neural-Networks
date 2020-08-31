# Leaky ReLU Activation Function 

Leaky ReLU is a variant of the [[ReLU]] activation function, used because it trains better than the ReLU function, and solves the [[Dying ReLU]] problem. 

The Leaky ReLU function has the formula:
$$\text{LeakyReLU}_\alpha(x) = \max(\alpha x, x)$$

It has been shown that a greater $\alpha$ (like 0.2) outperforms a smaller $\alpha$ (like 0.01).

and its graph is as follows:

<center>
	<iframe src="https://www.desmos.com/calculator/runwy8uyow?embed" width="500px" height="500px" style="border: 1px solid #ccc" frameborder=0></iframe>
</center>

### Why does Leaky ReLU fix the Dying ReLU problem?

There is no $x$ in the function such that $\frac{dy}{dx} = 0$. As a result, the neuron can never _die_. At worst, it can enter a _long coma_, after which it can recover.