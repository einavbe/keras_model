import time
import copy
start_time = time.time()
import numpy as np
import keras
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from keras.layers.core import Dense, Dropout, Activation, Flatten
import glob,os
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import initializers
from keras.regularizers import l2
from keras import backend as K
from sklearn.utils import shuffle
from tensorflow.python import debug as tf_debug
from model_functions1 import datalist, record, SampEn
import code
import csv
print('{}'.format(time.time()-start_time))


db_folder = r'C:\Users\Anna\Desktop\PCG RESEARCH\training\SQI Training'
folder_name = 'e'
datlist = datalist(db_folder)
db_folder = r'C:\Users\Anna\Desktop\Anna-VLS4U\Desktop\PCG RESEARCH\training\training-'

label = []
c_dat = np.empty((0, 4))

for index, row in datlist.iterrows():
    if index > 1782:
        file_name = row[0]
        lab = row[2]
        if lab is '':
            print('done')
            break

        else:
            crecord = record(file_name, db_folder + folder_name)
            if crecord.sample_entropy is None:
                continue
            c_dat = np.append(c_dat, [[crecord.sample_entropy, crecord.normalized_amplitude_envelope_kurtosis, crecord.max_correlation_peak, crecord.spectral_ratio]], axis = 0)
            label = np.append(label, [int(row[2])], axis=0)
            print("{} {} index is {}".format(folder_name, file_name, index))

    else:
        pass


c_dat = np.hstack((c_dat, np.reshape(label, [325, 1])))
#
os.chdir(r'C:\Users\Anna\Desktop\PCG RESEARCH\training\SQI Training')

with open('SQI.csv', 'a', newline='') as csvFile:
    # reader = csv.reader(csvFile)
    # lines = list(reader)
    # print(lines)
    for i in range (c_dat.shape[0]):
        writer = csv.writer(csvFile, delimiter = ' ')
        writer.writerow(c_dat[i])
    csvFile.close()

cdat = np.loadtxt('SQI1.csv')
# np.savetxt("SQI.csv", c_dat, fmt='%f', delimiter=' ')
# Z = np.loadtxt('SQI.csv')

# X_train = cdat[:, :4]
# X_train[:, 3] = np.log(X_train[:, 3])
# X_train = (X_train - np.mean(X_train, axis=0))/np.std(X_train, axis=0)
# Y_train = cdat[:, 4]
# Y_train = np_utils.to_categorical(Y_train, 4)
# Y_train = Y_train[:, 1:]
seed = 2
samp_size = round(0.8*(np.shape(cdat)[0]))
X = copy.copy(cdat[:, :4])
Y = cdat[:, 4]
X[:, 3] = np.log(X[:, 3])
X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
X,Y = shuffle(X, Y, random_state = seed)
X_train = X[:samp_size, :]
X_Val = X[samp_size:, :]

m = np.bincount((Y.astype(int)))[1:]
penal_class = {index: np.max(m)/numb for index, numb in enumerate(m)}

Y = np_utils.to_categorical(Y, 4)
Y_train = Y[:samp_size, 1:]
Y_Val = Y[samp_size:, 1:]


class get_grad(keras.callbacks.Callback):
    def __init__(self, x_train, y_train, model):
        self.x = x_train
        self.y = y_train
        self.model = model

    def on_epoch_end(self, epoch, logs={}):
        gradients = self.model.optimizer.get_gradients(self.model.total_loss, self.model.trainable_weights)
        input_tensors = self.model.inputs + self.model.sample_weights + self.model.targets + [K.learning_phase()]
        get_gradients = K.function(inputs=input_tensors, outputs=gradients)
        inpu = [self.x, np.ones(len(self.x)), self.y, 1]
        print('{}'.format(get_gradients(inpu)[0]))
        return


# class RMS_lr(keras.callbacks.Callback):
#     def __init__(self, i):
#         self.i = i
#
#     def on_train_end(self, logs=None):
#         opt = model.optimizer.lr
#         lr_val = K.eval(opt)
#         print('Learning rate: {}       {}/{}'.format(lr_val, self.i+1, 100))
#         return


a = 1

for i in range(10):
    if a+i < 3:
        print('yey')
    else:
        break



nb_epoch = 400

model = Sequential()
model.add(Dense(60, input_shape=(4,), kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(40, kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(20, kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(3, kernel_initializer='glorot_normal'))
model.add(Activation('softmax'))

model.summary()
    # if (load_weights):
    #     weights = model.load_weights(weights)


# for i in range(50):

    # lr = 10 ** np.random.uniform(0, -5)

loss = 'categorical_crossentropy'
optimizer = RMSprop(lr=0.001, rho=0.95, epsilon=1e-08, decay=1e-6)
# optimizer = SGD(lr=0.01, decay=1e-6)
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X_train, Y_train, validation_data=(X_train, Y_train), batch_size=128, epochs=nb_epoch, verbose=2, callbacks= [get_grad(X_train, Y_train, model)])


print('{0.7s}'.format('done'))



'''model_json = model.to_json()
with open("SQI_net_model", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(weights)
print("Saved model to disk")'''

# checkpointer = ModelCheckpoint(filepath=weights, verbose=1, save_best_only=True, save_weights_only=True)
# class_weight = {0: class_fact,
#                 1: 1}

# weights = model.trainable_weights
# gradients = model.optimizer.get_gradients(model.total_loss, weights)
# input_tensors = model.inputs + model.sample_weights + model.targets + [K.learning_phase()]
# get_gradients = K.function(inputs=input_tensors, outputs=gradients)
# inputs = [X_train, np.ones(len(X_train)), Y_train, 1]
# grads = get_gradients(inputs)
# , callbacks=[get_grad(X_train, Y_train, model)] - For gradient calculations per epoch




# def my_init(shape, dtype= None):
#     return k.
#
#
# model = Sequential()
# model.add(Dense(4, input_shape = (4,), kernel_initializer='random_normal'))
# # model.add(BatchNormalization())
# # model.add(Activation('relu'))
# # model.add(Dense(4, input_shape = (4,), kernel_regularizer = None))
# # model.add(BatchNormalization())
# # model.add(Activation('relu'))
# model.add(Dense(3,kernel_initializer='random_normal'))
# model.add(Activation('softmax'))
#
# # if (load_weights):
# #     weights = model.load_weights(weights)
#
# model.summary()
#
# model.compile(loss = loss, optimizer = optimizer , metrics = ['accuracy'])



#
# def SampEn(U, m, r):
# # Sample entropy SampEn has two advantages over ApEn
# # : data length independence and a relatively trouble-free implementation
# # Larger values of SampEn indicate a high variance signal ~ better PCG
#     def _maxdist(x_i, x_j):
#         return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
#
#     def _phi(m):
#         x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
#         C = [(len([1 for x_j in x if _maxdist(x_i, x_j) <= r])) - 1 for x_i in x]
#         return sum(C)
#
#     N = len(U)
#
#     return np.log(_phi(m)/_phi(m+1))
#
# U=range(10)
# m=2
# r=1
# import numpy as np
# Samp=SampEn(U,m,r)
# np.shape(U)
#
#
# t= np.arange(-10,10,0.0005)
# ttt = 10*np.sin(30*2*np.pi*t) + 7*np.sin(55*2*np.pi*t)
# z=np.fft.fft(ttt)
# c=z[0:int(np.size(z)/2)]
# ff = np.linspace(0,1000,np.size(c))
# plt.plot(ff, np.abs(c))
# N=4096
# f, Pxx = scipy.signal.welch(ttt, rate, 'hamming', nperseg = 4000, nfft = N, return_onesided=True)
# np.sum(Pxx[int(np.ceil(24 / fft_res_bin)):int(np.ceil(144 / fft_res_bin))]) / np.sum(Pxx[int(np.ceil(600 / fft_res_bin)):int(np.ceil(1000 / fft_res_bin))])
# _, Pxx = scipy.signal.welch(np.abs(c), rate, 'hamming', nperseg = 4000, nfft = N, return_onesided=True)
# plt.plot(f,Pxx)
#
#
#
# a = np.arange(1,1000)
# #b = scipy.signal.decimate(tt,2,ftype='iir',zero_phase=True)
# #d = scipy.signal.resample_poly(tt,1,2)
#
# N = 4096
# f, Pxx = scipy.signal.welch(ttt, 2000, 'hamming', nperseg=4000, nfft=N, return_onesided=True)
# fft_res_bin = 2000 / N
# np.sum(Pxx[int(np.ceil(24 / fft_res_bin)):int(np.ceil(144 / fft_res_bin))]) / np.sum(Pxx[int(np.ceil(600 / fft_res_bin)):int(np.ceil(1000 / fft_res_bin))])
#
# #def acf(sig):
#     #return ([np.corrcoef (sig[:-i], sig[i:])[0,1].astype(float) for i in range(1, sig.size)])
#
# from IPython import embed
# embed()
# import timeit
#
#
# timeit.timeit('''
# import numpy as np
# b = np.ones ((3,50,50))
# a = np.empty((2999,3,50,50))
# a = np.append (a,[b], axis=0)
# ''', number=1)
#
#
# timeit.timeit('''
# import numpy as np
# a = np.zeros ((3000,3,50,50))
# ''', number=1)
#
# import numpy as np
# a=np.empty(3000,3,50,50)
# a=np.zeros(3000,3,50,50)
#
#
#
# import numpy as np
# b = np.ones ((3,50,50))
# a = np.empty((10,3,50,50))
# a[0]=b
#
#
# a = np.append(a,[b],axis=0)
#
#
# timeit.Timer('import numpy as np a=np.zeros((10,3,50,50))').timeit()



# import scipy.signal
# import scipy.io.wavfile as wav
# from scipy.signal import hilbert
# from sampen import sampen2
#
#
#
# whole_name = r'C:\Users\Anna-VLS4U\Desktop\PCG RESEARCH\training\training-a'
# file_name = 'a0001'
# whole = whole_name + '\\' + file_name  +  '.wav'
# rate, sig = wav.read(whole)
# hilbert_envelope = hilbert(sig, N=int(sig.shape[-1] / 1))
# amplitude_envelope = np.abs(hilbert_envelope)
# normalized_amplitude_envelope = (amplitude_envelope - np.mean(amplitude_envelope)) / np.std(amplitude_envelope)
# auto_corr_sig = np.correlate(normalized_amplitude_envelope, normalized_amplitude_envelope, mode='full')
# trunc_sig = int(np.floor(auto_corr_sig.size / 2))
# auto_corr_sig = auto_corr_sig[trunc_sig:]
# auto_corr_sig_norm = (auto_corr_sig - np.mean(auto_corr_sig)) / np.std(auto_corr_sig)
# first_zero_cross_idx = (np.where(np.diff(auto_corr_sig_norm) > 0))[0][0] + 1
# auto_corr_sig_norm1 = auto_corr_sig_norm[first_zero_cross_idx:first_zero_cross_idx+5*rate if first_zero_cross_idx+5*rate<total_length*rate else np.size(auto_corr_sig_norm)]
#
# import time
# sampen2(auto_corr_sig_norm1, 2, 0.0008)[2][1]
#
#
# start_time = time.time()
#
# SampEn(auto_corr_sig_norm1, 2, 0.0008)
# print('time elapsed {:f}'.format(time.time()-start_time))
#
#
#
#
#
#
#
#
#
#
# def SampEn(U, m, r):
# # Sample entropy SampEn has two advantages over ApEn
# # : data length independence and a relatively trouble-free implementation
#     def _maxdist(x_i, x_j):
#         return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
#
#     def _phi(m):
#         x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
#         C = [(len([1 for x_j in x if _maxdist(x_i, x_j) <= r])) - 1 for x_i in x]
#         return sum(C)
#
#     N = len(U)
#
#     return np.log(_phi(m)/_phi(m+1))
#
#
# timeit.timeit('''
# import numpy as np
# a = np.ones(10)
# b = np.array([3,6,2,4,6,2,6,4,8,1])
# print((b-a).max())
# ''', number = 1)
#
#
#
# a = np.ones(10)
# b = np.array([3,6,2,4,6,2,6,4,8,1])
# print((b-a).max())


from functools import wraps

def logit(func):
    @wraps(func)
    def with_logging(*args, **kwargs):
        print(func.__name__ + " was called")
        return func(*args, **kwargs)
    return with_logging

@logit
def addition_func(x):
   """Do some math."""
   return print(x + x)


result = addition_func(4)




