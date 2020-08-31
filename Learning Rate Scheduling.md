# Learning Rate Scheduling

The learning rate is one of the most important hyperparameters to be tuned. 

If we set it too high, training can diverge. But if we set it too low, training will take very long. Also if we set it slightly too high, training can just dance around the optimum.
<br>

![[Learning Rates Curve.png]]

If we start with a large learning rate (to traverse the landscape quickly) and then reduce it when training slows down (to make smaller adjustments), we reach the optimal solution much quicker.

This is called _Learning rate scheduling_.


#### Commonly used schedules:
- [[Power Scheduling]]
- [[Exponential Scheduling]]
- [[Piecewise Constant Scheduling]]
- [[Performance Scheduling]]
- [[1 Cycle Scheduling]]