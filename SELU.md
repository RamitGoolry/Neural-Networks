# SELU Activation Function
SELU stands for Scaled [[ELU]].

In their 2017 [Paper](https://arxiv.org/abs/1706.02515), Klambauer et al. introduced this activation function. They showed that if you build a [[Neural Network]] composed exclusively of `Dense` layers and then use the SELU function, the network will _self normalize_, i.e., the outputs will preserve a mean of 0 and a standard deviation of 1 during training, which solves the [[Vanishing and Exploding Gradients problem]].