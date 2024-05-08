from sklearn.preprocessing import normalize
import numpy as np

def loadGraph(args):
    adjgat = np.load(args.adjgat_file)
    return adjgat