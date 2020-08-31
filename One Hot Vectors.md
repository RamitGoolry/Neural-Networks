# One Hot Vectors
One Hot Vectors are generally used when we have a few categorical features, which have to be used (less than 10). It lets Neural Networks easily distinguish between various categories with ease. 

They are very easy to create:

```python

vocab = ['''Categories''']
indices = tf.range(len(vocab), dtype = tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
num_oov_buckets = 2
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)

```

###### Q : What are oov buckets?
Out of index buckets are used whenever we might come across new data during implementation or just messy data. The more unknown categories you expect to find, the more oov buckets you should use or there will be collisions.