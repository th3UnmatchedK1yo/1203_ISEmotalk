import numpy as np
import keras
import tensorflow
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint                        
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical


