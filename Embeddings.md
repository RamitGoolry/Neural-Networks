# Embeddings

An embedding is a trainable Dense vector that represents a category. Embeddings are given a dimension, and are initialized randomly. Since they are trainable, [[Gradient Descent]] will end up pushing them in some kind of structure, which has some meaning associated to them.

As a result, training tends to make embeddings useful representations of the categories. This is called [[Representation Learning]].

### Word Embeddings
Word Embeddings see widespread use in Language models. They are semantic representations of words, and their structure holds meaning.

> You are better off using pre trained word embeddings.

#### Structure of Word Embeddings
- __Distance__ - Semantically related words such as France, Spain and Italy end up clustering together.
- __Axis__ - Word Embeddings are also organized along meaningful axes : such that they make sense in relation. A common example is 
 $$\text{King} - \text{Man} + \text{Woman} \approx \text{Queen} $$
 
 ## Implementation
 Keras provides a `keras.layers.Embedding` layer that handles the embedding matrix.