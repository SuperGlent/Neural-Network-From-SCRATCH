from abc import ABC, abstractclassmethod


"""
All classes has to implement __call__ function!
"""

class IActivationFunctions(ABC):
    
    @abstractclassmethod
    def __init__(self):
        super().__init__()
    
class ILayer(ABC):
    
    @abstractclassmethod
    def __init__(self):
        super().__init__()
        
    @abstractclassmethod
    def __call__(self):
        pass
    
    @abstractclassmethod
    def show_params(self):
        pass