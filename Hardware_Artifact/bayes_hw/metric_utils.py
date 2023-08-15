import numpy as np 

def entropy(output):
  batch_size = output.shape[0]
  entropy = -np.sum(np.log(output+1e-8)*output)/batch_size
  return entropy