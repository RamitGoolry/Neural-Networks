# AdaGrad

Think about the [[Gradient Descent]] slope again:

![[Gradient Descent Slope.jpg]]

[[Gradient Descent]] starts by quickly going down the steepest slope, which in most cases does not point to the global optimum and then very slowly goes down to the bottom of the valley.

AdaGrad seeks to improve on this problem by correcting the direction of the gradient earlier to point closer to the global optimum. It does this by scaling down the gradient vector along the steepest dimensions.

Basically, the algorithm decays the learning rate, but it does so faster for steeper dimensions than for dimensions with steeper slopes. 

But herein also lies a __problem__. AdaGrad stops too early when training [[Neural Network]]s. The learning rate is scaled down so much, that the algorithm ends up stopping entirely before reaching the global optimum. This is why, even though Keras does support an `AdaGrad` optimizer, it's not recommended. 