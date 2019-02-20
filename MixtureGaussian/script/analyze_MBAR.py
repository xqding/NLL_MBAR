import pickle
import numpy as np

MBAR_F = []
for i in range(50):
    with open("./output/MBAR_F_{}.pkl".format(i), 'rb') as file_handle:
        MBAR_F.append(pickle.load(file_handle))
        
F = np.array(MBAR_F)

F_mean = np.mean(F[:,-1])
F_std = np.std(F[:,-1])

print("MBAR result: {:.3f} +- {:.3f}".format(F_mean, F_std))


        
