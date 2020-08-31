# Unsupervised Pretraining
Unsupervised pretraining is another way to increase training speed and accuracy, if you have little (labeled) training data to work with. The idea is that you can used unlabeled training data, which is rather easy to find and use that to train a [[Autoencoder]] or a [[Generative Adversarial Network]].

You can then [[Transfer Learning|transfer]] the lower layers of the [[Autoencoder]] or the lower layers of the [[Generative Adversarial Network|GAN]]'s discriminator.

This will allow for good feature extraction even if we don't habe too much data.

### Example : Facial Recognition

If we want to make a classifier that recognizes faces, but we only have a few faces to classify, we can gather lots of random faces off the web and then train a [[Neural Network]] to detect whether or not 2 pictures have the same person (using something like [[Triplet Loss]]). Reusing lower layers of this network will then work as good feature detectors.