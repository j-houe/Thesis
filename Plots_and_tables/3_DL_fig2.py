import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import ReLU, Sigmoid, Softplus

####################################
# Plot common activation functions #
####################################
x = torch.Tensor(np.linspace(-5, 5, 100))
relu = ReLU()
sigmoid = Sigmoid()
softplus = Softplus()
fig, axs = plt.subplots(1, 3)
axs[0].plot(x, relu(x))
axs[0].set_title('ReLU')
axs[1].plot(x, softplus(x))
axs[1].set_title('SoftPlus')
axs[2].plot(x, sigmoid(x))
axs[2].set_title('Sigmoid')
fig.set_size_inches(8, 2.5)
plt.show()
