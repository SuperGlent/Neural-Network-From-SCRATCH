from main import ActivationFunctions, Layer, Dense, Conv2D, Pooling, Flatten, Dropout, BatchNormalization, Input, SeqNet, Loss, BackProp, Optimizers, Adam, NN
import unittest
import numpy as np

functions = {
"act_func_linear": ActivationFunctions(),
"act_func_sigmoid": ActivationFunctions("sigmoid"),
"act_func_relu": ActivationFunctions("relu"),
"act_func_tanh": ActivationFunctions("tanh")
}

x = np.random.random(size=(10, ))
print(f"X is: {x}, with shape: {x.shape}")
for i, func in functions.items():
    print(func)
    print(f"Function {i} output from sample \n {x} \n {func(x)}")