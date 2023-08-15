import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class _DropoutBase(nn.Module):
    r"""
    This is a base model for all Bayesian dropout models.
    """
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, layer: nn.Module, p: float = 0.5, inplace: bool = False) -> None:
        super(_DropoutBase, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.layer = layer 
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)

class BayesianDropout(_DropoutBase): 
    r"""
     This class is similar to pytorch nn.Dropout but the training value is always 
     set to True in accordance to the behaviour of Bayesian Neural Network. This 
     might be an abuse of the semantics. A given layer is also used before applying
     dropout.
    """
    def forward(self, input):
      input = self.layer(input)
      return F.dropout(input, self.p, True, self.inplace)

class BayesianDropout2D(_DropoutBase): 
    r"""
     This class is similar to pytorch nn.Dropout but the training value is always 
     set to True in accordance to the behaviour of Bayesian Neural Network. This 
     might be an abuse of the semantics. A given layer is also used before applying
     dropout.
    """
    def forward(self, input):
      input = self.layer(input)
      return F.dropout2d(input, self.p, True, self.inplace)

class BayesianDropout3D(_DropoutBase): 
    r"""
     This class is similar to pytorch nn.Dropout but the training value is always 
     set to True in accordance to the behaviour of Bayesian Neural Network. This 
     might be an abuse of the semantics. A given layer is also used before applying
     dropout.
    """
    def forward(self, input):
      input = self.layer(input)
      return F.dropout3d(input, self.p, True, self.inplace)