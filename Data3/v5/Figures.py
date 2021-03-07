import matplotlib.pyplot as plt
import pickle
import numpy as np

pickle.load(open('pickle/roc.fig.pickle', 'rb'))
pickle.load(open('pickle/auc.fig.pickle', 'rb'))
pickle.load(open('pickle/loss.fig.pickle', 'rb'))
plt.show()
