import numpy as np
import matplotlib.pyplot as plt
from INet import ILayer

#helpers
def require(proc, mes):
    if not proc:
        print(mes)
        raise ValueError

"""
Activation Functions class, takes as argument activ function for layer. 
As Initializing it set up activation function. 
Then, to have an output - use func()
Every function returns np.ndarray, as calculating activation function for whole the layer!
Has to be used as init object and giving it to a layer.
"""
class ActivationFunctions:
    def __init__(self, activation_func="linear"):
        match activation_func:
            case "linear":
                self.activation_func = self.linear
            case "sigmoid":
                self.activation_func = self.sigmoid
            case "tanh":
                self.activation_func = self.tanh
            case "relu":
                self.activation_func = self.relu
        #return self.activation_func

    #service string representation
    def __repr__(self):
        print(f"""Object of class ActivationFunctions, current activation function: {self.activation_func}. 
              Every function returns np.ndarray, as calculating activation function for whole the layer!
              Has to be used as init object and giving it to a layer.""")

    def linear(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def sigmoid(self, inputs: np.ndarray) -> np.ndarray:
        return np.devide(np.ones(shape=inputs.shape), (np.exp(inputs) + 1), data_shape=inputs.shape) #for i in input.tolist()])

    def tanh(self, inputs: np.ndarray) -> np.ndarray:
        return np.devide((np.exp(inputs) - np.exp(-inputs)), (np.exp(inputs) + np.exp(-inputs)))

    def relu(self, inputs: np.ndarray) -> np.ndarray:
        for i, v in enumerate(inputs.tolist()):
            if v <= 0:
                input[i] = 0
            else:
                input[i] = v

    def softmax(self, inputs) -> np.ndarray:
        exp_weigth = np.exp(inputs)
        return np.devide(exp_weigth, exp_weigth.sum(), data_shape=inputs.shape)

    def __call__(self, x) -> np.array:
        return self.activation_func(x)

"""
Simple layer class, inherits from layer abstract class.
When init - creates parametrs, init weigth.
As well all layers has to inherits from ILayer, because they all has to implement 3 functions - __init__, __call__, show_params.
This realization of layer gives as output weigthed sum of all inputs.
Created to inherit from it sublasses as Dense, Conv... all that gives as output something like weigthed sum or array.
Has to be updated to create more general behavior
"""
class Layer(ILayer):

    def __init__(self, input_shape: int, output_shape: int, activation: ActivationFunctions):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weigth = self.weigth_init() #weigth init for a layer as initializing them like what is coming from input, so, PREVIOUS. + bias
        self.activation = activation #there we will define certain activation func, to take value call it.
        self.outputs = np.zeros(output_shape) #have no idea why, probably useless.

    #has to return layer parametrs for longer inplementation to show net function
    def show_params(self) -> dict:
        return {"activation": self.activation, 
                "shape": (self.input_shape, self.output_shape), 
                "weigth": self.weigth}

    #When calling returns a output vector, calculated as dot prod of layer weigth with inputs, with dimension output_shape. INCLUDES BIAS!
    def __call__(self, data: np.ndarray, to_write=True) -> tuple:
        try:
            outputs = self.weigth @ np.append(data, [1]) # + bias
            outputs = self.activation(outputs)
            if to_write:
                self.outputs = outputs
            return outputs, self.output_shape
        
        except ValueError as error:
            print(f"Catched an error: {error}, make the right array shapes!")
            
    def weigth_init(self, method="random", range=[-0.2, 0.2]):
        match method:
            case "random":
                return np.random.rand(self.output_shape+1, self.input_shape)
            case "scalar":
                require(range==float, "Range has to be a float")
                return np.ones((self.output_shape+1, self.input_shape)) * range
            case "Xavier":
                pass
            case "He":
                pass


#Layer subclass, just a basic Layer
class Dense(Layer):
    def __init__(self):
        super(self).__init__()


#2 Dimensional Convolutional layer
class Conv2D(Layer):
    def __init__(self, kernels: int, strides: int, same_shape=True):
        pass


#Pooling layer for conv nets
class Pooling(ILayer):
    
    def __init__(self, pooling_type="maxPooling"):
        match pooling_type:
            case "maxPooling":
                pass
            case "avPooling":
                pass
            case "minPooling":
                pass
    
    def __call__(self, input):
        pass


#Creates 1 dim array from every dim array.
class Flatten(ILayer):
    def __init__(self):
        pass
    
    
#Randomly turns off certain percent of neurons for the batch
class Dropout(ILayer):
    def __init__(self, turnoff_rate=0.2):
        pass
    
    
#Normalize batch 
class BatchNormalization(ILayer):
    def __init__(self):
        pass
    

#point that is not a Layer subclass, it has only to make the data right format and stand for visual comfort.
class Input(ILayer):
    
    def __init__(self, data_shape=None):
        self.data_shape = data_shape
        
    #returns what you input and shape of it for init input shape of first real layer (Dense as example)
    def __call__(self, input_data) -> tuple:
        self.data_shape = input_data.shape
        return input_data, self.data_shape
    
    #To have a simple implementation in main algorithm
    def show_params(self) -> None:
        return None
    

#Class for creating simple Sequential network with layers you input one by one.
class SeqNet:

    #Layers: {Input: (), Dense:  (output_shape, activation), Dense: (output_shape, activation)}
    def __init__(self, layers: dict):
        self.net = [layer(params) for layer, params in layers]
        
    #Returns parametrs for every layer - for optimizing parametrs. See Optimizer
    def get_params(self) -> dict:
        return {i: v.show_params() for i, v in enumerate(self.net)}


#Class for compute losses. Also used like a storage for all losses (for graphics as well). POTENTIALY REMAKE!!! CREATE MSE AND CE CLASSES.
class Loss:
    
    def __init__(self, loss="MSE", params=None): #-> function
        self.episodes_loss = []
        self.loss = loss
        
        match loss:
            case "MSE":
                return self.MSE
            case "CE":
                return self.CE
            
    #SERVICE FUNCTIONS:
    #clean episode loss
    def clean_losses(self) -> int:
        print("Updating losses list...")
        self.episodes_loss = []
        return 0

    #returns loss type and all losses
    def show_params(self) -> dict:
        return {
            "loss": self.loss, 
            "lossForEpisodes": self.episodes_loss
                }

    #service string representation
    def __repr__(self):
        print(f"""This is object of a Loss class. It has to calculate mean loss for BATCH, so it always returns scalar.
              We don't actually need it when training, so gradients are calculating derivative of it.
              When calling, batch expected!""")

    #LOSSES
    #Mean squered error
    def MSE(self, Y_pred, Y_true) -> float:
        require(Y_pred.shape==Y_true.shape, "Incorrect output shapes")
        loss = np.mean((Y_true - Y_pred) ** 2)
        self.episodes_loss.append(loss)
        return loss

    #Cross Enthropy
    def CE(self, Y_pred, Y_true) -> float:
        require(Y_pred.shape==Y_true.shape, "Incorrect output shapes")
        loss = np.mean(np.sum(- Y_pred * np.log(Y_pred), axis=1))
        self.episodes_loss.append(loss)
        return loss
    

"""
One of the most important classes. 
Computes all gradients for all layers.
Do not use optimizer in there. Just returns gradients per layer.
"""
class BackProp:
    def __init__(self, loss: str, net: SeqNet):
        self.loss = loss
        self.net = net

    #What is a derivative of activation function?
    @staticmethod
    def derivative(x, activ) -> float:
        match activ:
            case ActivationFunctions.relu:
                if x:
                    return 1.
            case ActivationFunctions.tanh:
                pass

            case ActivationFunctions.softmax:
                pass

    #Service string representation of class
    def __repr__(self):
        print(f"""The main method for calculating gradients. Returns all layers gradients""")

    #service function which COMPUTE gradients for every layer by chain rool (back propogation)
    def layers_grads(self, Y_true, Y_pred) -> list:
        batch_size, y_size = Y_true.shape
        layer_grads = []
        layers_activ = [activ for activ in self.net.activation]
        layers_outputs = [output for output in self.net.outputs].pop(-1)
        match self.loss:
            case Loss.MSE:
                for i, layer in self.net.layers: #so suppose to be a dict
                    if i == 0:
                        dLdfx = 2 * Y_true - Y_pred
                        dfxdv = self.derivative(layers_outputs[-i+1], layers_activ[-i+1]) #v is dot product xw + xw + xw + b
                        grad = np.mean(dLdfx * dfxdv * layers_outputs[-1].T, axis=0)
                    else:
                        rec_layer_params = self.net.layers[-i+1]
                        grad = layer_grads[-1] * self.derivative(layers_outputs[-i+1], layers_activ[-i+1]) * rec_layer_params
                        layer_grads.append(grad)
        return layer_grads.reversed()

    #RETURNS all the gradients for all the layers. 
    def __call__(self, *args, **kwargs) -> np.ndarray:
        Y_true, Y_pred = kwargs.values()["Ys"]
        return np.array(self.layer_grads(self.net, Y_true, Y_pred))


"""
This class created to optimize gradient descent algorithm different way. Exactly this one make weigth change.
Add a few other optimizers for experiments.
When initializing set up giperparametrs like learning rate and other general for most of optimizers. (have to add)
When calling returns learning iteration by calling setted optimizer.
"""
class Optimizers:
    def __init__(self, net: SeqNet, default="Adam", lr=0.001):
        if default == "Adam":
            self.optimiser = self.Adam
        self.lr = lr
        self.gardient_memory = []
        self.current_grads = []

    def Adam(self):
        return self.current_grads
    
    def __call__(self, gradients, model_params):
        self.optimiser(gradients, model_params)
      
      
class Adam:
    
    def __init__(self, params_shape, learning_rate=0.001, beta_momentum=0.9, beta_RMS=0.999):
        self.lr = learning_rate
        self.bm = beta_momentum
        self.br = beta_RMS
        params_shape = params_shape
        self.layers_m = np.zeros(params_shape)
        self.layers_v = np.zeros(params_shape)
        
    def calc_m(self, layer_m, layer_grad: np.ndarray):
        mt = self.bm * layer_m + (1-self.bm) * layer_grad
        return mt / (1 - self.bm) #bias normalized
    
    def calc_v(self, layer_v, layer_grad: np.ndarray):
        vt = self.br * layer_v + (1-self.br) * layer_grad**2
        return vt / (1 - self.br) #bias normalized
    
    def __call__(self, gradients: list, model_params: np.ndarray):
        optim_grads = [layer_grad for layer_grad in gradients]
        [model_params[layer] for layer in range(model_params.shape[0])]
        return 
  
        
"""
Main class which creates model.
When initializing: creates giperparametrs as epochs, model name, batch size, data samples count(train_size).
Takes as arg SeqNet.
Graphs, Loss, Optimizers, BackProp classes are implemented there.
Use fit, evaluate, getModelInfo methods.
"""
class NN:

    def __init__(self, model_name: str, train_size: tuple, net: SeqNet, epochs=1000, batch_size=50, optimizer="Adam", loss="MSE"):
        #params
        self.net = net
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_size = train_size
        
        #losses
        self.loss = Loss(loss)
        self.val_loss = Loss(loss)
        self.test_loss = Loss(loss)
        
        #Classes algos
        self.optim = Optimizers(optimizer)
        self.back_prop = BackProp(self.loss, self.optim, self.net)
        self.losses_graph = Graphs()
        
        #Accuracys
        self.accuracy = []
        self.test_accuracy = []

    #calculate model accuracy
    def calc_accuracy(self, Y_true, Y_pred):
        pass
        
    #fit model, creates graohics automaticly
    def fit(self, X_train, Y_train) -> None:
        for epoch in self.epochs:
            print(f"Epoch {"-"*20} {epoch} {"-"*20}")
            #SPLIT DATA TO VALIDATION AND TRAIN!!
            for _ in range(0, self.train_size[0], self.batch_size):
                X_train_batch = np.array(X_train.tolist()[0:self.batch_size])
                Y_train_batch = np.array(Y_train.tolist()[0:self.batch_size])
                for _ in range(self.batch_size):
                    Y_preds = self.forward(X_train_batch)
                    gradients = self.back_prop(Ys=(Y_preds, Y_train_batch))
                self.optim(gradients, self.net.get_params())
        self.garph.show(self.loss.episodes_loss, self.val_loss.episodes_loss)
        
    #test model on unseed data
    def evaluate(self, X_test, Y_test) -> None:
        for pred in X_test.shape[0]:    
            Y_pred = self.net(X_test)
            loss = self.test_loss(Y_test, Y_pred)
            self.test_accuracy += self.calc_accuracy(Y_test, Y_pred)
        print(f"Test predictions count: {X_test.shape[0]} {"-"*20} Test Accuracy: {self.test_accuracy} {"-"*20} Test mean loss: {sum(self.test_loss.episodes_loss)/len(self.test_loss.episodes_loss)}")
        self.graph.show(self.val_loss.show_params()["episode_loss"])
            
    #service function for forward pass on model
    def __forward(self, data: np.ndarray) -> np.ndarray:
        if data.shape == "odnomernyi massiv":
            return self.net(data)
        elif data.shape == "mnogomerny massiv":
            return [self.net(np.array(sample)) for sample in data.tolist()]
        
    #Returns a dictionary with model parametrs info.
    def getModelInfo(self) -> dict:
        return {"ModelName": self.model_name, 
                "ModelConfig": self.model_net,
                "TrainEpochs": self.epochs,
                "BatchSize": self.batch_size,
                "Loss": self.loss.show_params()["loss"],
                "Optimizer": self.optim.optimiser}

#For creating graphics
class Graphs:
    def __init__(self):
        pass

#ALGORITHM simple visualization.
'''data_shape = None
net = SeqNet(
    {
        Input: None, 
        Dense: (128, "relu"), 
        Dropout: 0.2,
        BatchNormalization: None,
        Dense: (64, "relu"),
        Dropout: 0.2,
        BatchNormalization: None,
        Dense: (1, "linear")
              }
            )

model = NN(model_name="ReallyFirstModel", train_size=data_shape, net=net)

print(model.fit(X_train=None, Y_train=None))
print(model.getModelInfo())
print(model.evaluate(X_test=None, Y_test=None))
'''