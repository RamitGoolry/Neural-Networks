# Difference Between Metrics and Losses

Metrics and Losses are conceptually different. While the same functions can be used for the both of them, they serve very different functions.

Losses are used by [[Gradient Descent]] to _train_ a model. They must be differentiable.

Metrics are used to _evaluate_ a model. They must be easily interpretable.