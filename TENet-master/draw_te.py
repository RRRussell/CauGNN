import matplotlib as plt

import matplotlib.pyplot as plt
import numpy as np
file = '/home/jiangnanyida/Documents/MTS/MTS_TEGNN/TENet-master/TE/ente.txt'
A = np.loadtxt(file)
A = np.array(A,dtype=np.float32)

# Display matrix
plt.matshow(A)

plt.show()
