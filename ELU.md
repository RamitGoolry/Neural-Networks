# ELU Activation Function
ELU (Exponential Linear Unit) is a variant of the [[ReLU]] activation function, and solves the [[Dying ReLU]] problem, because of its non-linear, non-saturating nature.

The ELU function has the formula:
$$\text{ELU}_\alpha(x) = 
\begin{cases}
	\alpha (e^x - 1) & \text{if } x < 0 \\
	x & \text{otherwise}
\end{cases}$$

The value $\alpha$ defines the value the graph approaches as its $x$ approaches a large negative number

and its graph is as follows:
<center>
	<iframe src="https://www.desmos.com/calculator/etzunhoudd?embed" width="500px" height="500px" style="border: 1px solid #ccc" frameborder=0></iframe>
</center>

#### Pros
- It takes on negative values when $x < 0$, so its mean is closer to 0. This helps fix the [[Vanishing and Exploding Gradients problem]].
- It has a nonzero differential for $x < 0$, so it doesn't face the [[Dying ReLU]] problem.
- When $\alpha = 1$, then the function is smooth everywhere, speeding up [[Gradient Descent]].

#### Cons
- The ELU function is really slow to compute, but it's faster convergence lets makes up for it. However, ELU is slower than [[ReLU]] during testing.
