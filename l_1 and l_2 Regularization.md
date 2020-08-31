# l_1 and l_2 Regularization
#TODO

We can use $\ell_2$ regularization to constrain a [[Neural Network]]'s connection weights and/or $\ell_1$ regularization if we want a sparse model. 

## Usage 
```python
layer = keras.layers.Dense(100, 
			activation = 'elu', 
			kernel_initializer = 'he_normal',
			kernel_regularizer = keras.regularizers.l2(0.01)
)
```

We can also use `keras.regularizers.l1()` if we want to use $\ell_1$ regularization or `keras.regularizers.l1_l2()` if we want to use both.