import h5py
h5py.run_tests()                                   
                                                                                             
import numpy as np


h5 = h5py.File("MA0035_4_subst_train.h5", "r")
                                                                                             
data = h5["data"]                          
                                                                                             
print(data.shape)                                                      
# Out[5]: (100000, 4, 1000)                                             

labels = h5["labels"]                     

labels = labels[:]

print(labels.shape)

print(data[1])                                    
print(labels[1])
                                                                                             
# print(np.sum(labels > 0.1))                          
# Out[8]: 51790                                
                                                                                             
# print(np.sum(labels > 100))                
# Out[9]: 45758                                
                                                                                             
# print(np.max(labels))                        
# Out[10]: 27800.0                                    
                                                                                             
# print(np.sum(labels > 1000))                          
# Out[11]: 24988


