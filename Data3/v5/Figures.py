import matplotlib.pyplot as plt
import pickle
import numpy as np

root = 'logs/1_layer_dropout/log30'

pickle.load(open(root + '/pickle/roc.fig.pickle', 'rb'))
pickle.load(open(root + '/pickle/auc.fig.pickle', 'rb'))
pickle.load(open(root + '/pickle/loss.fig.pickle', 'rb'))
plt.show()
