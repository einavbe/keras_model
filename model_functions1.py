from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
#import hbdata_predict_v2 as dataimport
from keras.models import model_from_json
import argparse
#from __future__ import absolute_import
from os import listdir
from os.path import isfile, join
import numpy as np
import itertools
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import scipy.signal
from numpy import inf
import random
import math
import sys
import scipy.stats as scistat
from scipy.signal import hilbert
#sys.path.insert(0, 'C:/Users/Anna-VLS4U/Desktop/PCG RESEARCH/python_speech_features-master')
#mfcc(np.array([1.2]))
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sampen import sampen2

def ApEn(U, m, r):
#Approximate_entropy
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m+1) - _phi(m))

def SampEn(U, m, r):
# Sample entropy SampEn has two advantages over ApEn
# : data length independence and a relatively trouble-free implementation
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [(len([1 for x_j in x if _maxdist(x_i, x_j) <= r])) - 1 for x_i in x]
        return sum(C)

    N = len(U)

    return np.log(_phi(m)/_phi(m+1))

def seSQI (sig, rate, total_length):

    hilbert_envelope = hilbert(sig, N=int(sig.shape[-1] / 1))
    amplitude_envelope = np.abs(hilbert_envelope)
    # instantaneous_phase = np.unwrap(np.angle(hilbert_envelope))
    # instantaneous_frequency = (np.diff(instantaneous_phase) /(2.0 * np.pi) * rate)
    normalized_amplitude_envelope = (amplitude_envelope - np.mean(amplitude_envelope)) / np.std(amplitude_envelope)

    # Calculating the autocorrelation of the envelope, normalizing it, and removing the peak at Lag [0] - from David springer "Automated signal quality assessment..."
    auto_corr_sig = np.correlate(normalized_amplitude_envelope, normalized_amplitude_envelope, mode='full')
    trunc_sig = int(np.floor(auto_corr_sig.size / 2))
    auto_corr_sig = auto_corr_sig[trunc_sig:]
    auto_corr_sig_norm = (auto_corr_sig - np.mean(auto_corr_sig)) / np.std(auto_corr_sig)
    first_zero_cross_idx = (np.where(np.diff(auto_corr_sig_norm) > 0))[0][0] + 1
    auto_corr_sig_norm1 = auto_corr_sig_norm[first_zero_cross_idx:first_zero_cross_idx+5*rate if first_zero_cross_idx+5*rate<total_length*rate else np.size(auto_corr_sig_norm)]

    # Calculating the Sample Entropy
    return sampen2(auto_corr_sig_norm1, 2, 0.0008)[2][1], normalized_amplitude_envelope,auto_corr_sig_norm

def mSQI (sig, rate, total_length): #Downsample by 2 for now since physionet signals are 2K and we need 1KHz
    down_samp_sig = scipy.signal.decimate(sig, 2, ftype='iir', zero_phase=True)
    rate = rate//2
    hilbert_envelope = hilbert(down_samp_sig, N=int(down_samp_sig.shape[-1] / 1))
    amplitude_envelope = np.abs(hilbert_envelope)
    normalized_amplitude_envelope = (amplitude_envelope - np.mean(amplitude_envelope)) / np.std(amplitude_envelope)

    # Calculating the autocorrelation of the envelope, normalizing it, and removing the peak at Lag [0] - from David springer "Automated signal quality assessment..."
    auto_corr_sig = np.correlate(normalized_amplitude_envelope, normalized_amplitude_envelope, mode='full')
    trunc_sig = int(np.floor(auto_corr_sig.size / 2))
    auto_corr_sig = auto_corr_sig[trunc_sig:]
    auto_corr_sig_norm = (auto_corr_sig - np.mean(auto_corr_sig)) / np.std(auto_corr_sig)
    first_zero_cross_idx = (np.where(np.diff(auto_corr_sig_norm) > 0))[0][0] + 1
    auto_corr_sig_norm = auto_corr_sig_norm[first_zero_cross_idx: first_zero_cross_idx + 5 * rate if first_zero_cross_idx + 5 * rate < total_length * rate else np.size(auto_corr_sig_norm)]

    # Finding the maximum peak in the autocorrelation between Lag 500 and 2330 samples (for 1Khz sample rate correlates to 0.5-2.33s) - from same article as in seSQI function
    return np.max(auto_corr_sig_norm[500:2330])

def sdrSQI (sig, rate):
    N = 4096
    f, Pxx = scipy.signal.welch(sig, rate, 'hamming', nperseg = 4000, nfft = N, return_onesided=True)
    fft_res_bin = rate/N
    return np.sum(Pxx[int(np.ceil(24 / fft_res_bin)):int(np.ceil(144 / fft_res_bin))]) / np.sum(Pxx[int(np.ceil(600 / fft_res_bin)):int(np.ceil(1000 / fft_res_bin))])

def bSQI (sig):
    pass # Not sure if to implement or not

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class features_setting(object):
    __metaclass__ = Singleton
    def __init__(self,length,width, samplerate, winlen, winstep, numcep,
                          nfilt, nfft, lowfreq, highfreq, preemph, ceplifter,
                          appendEnergy):
        self.length=length
        self.width=width
        self.samplerate = samplerate
        self.winlen = winlen
        self.winstep = winstep
        self.numcep = numcep
        self.nfilt = nfilt
        self.nfft = nfft
        self.lowfreq = lowfreq
        self.highfreq =highfreq
        self.preemph = preemph
        self.ceplifter = ceplifter
        self.appendEnergy = appendEnergy
        self.total_window_length=winlen+length*winstep
class model_setting(object):
    '''
        :keyword

    '''
    __metaclass__ = Singleton
    def __init__(self, load_weights, model_name,weights,nb_filters,nb_classes,nb_hidden,loss,optimizer,nb_epoch,batch_size,class_fact):
                self.load_weights=load_weights
                self.model_name=model_name
                self.weights = weights
                self.nb_filters= nb_filters
                self.nb_classes=nb_classes
                self.nb_hidden=nb_hidden
                self.loss = loss
                self.optimizer = optimizer
                self.nb_epoch=nb_epoch
                self.batch_size=batch_size
                self.class_fact=class_fact




class record(object):
    def __init__(self, name, folder_name):
        """creating the record micro structure for every record, by giving 2 parameters which are the file name and location,
        the given file is opened used initialize parameter function, the latter, extract:
        signal_length - the length of the signal [Seconds]
        sampling_rate - the file sampling rate 2K hz for PHysionet DB

        """
        self.name = name
        self.folder_name = folder_name
        self.whole_name = folder_name+'\\'+name+'.wav'
        self.total_length=9
        self.snr=None
        self.sampling_rate=None
        self.normalized_amplitude_envelope_variance=None
        self.normalized_amplitude_envelope_kurtosis=None
        self.signal_kurtosis = None
        self.max_correlation_peak=None
        self.sample_entropy=None
        self.initialize_parameters()


        # self init should be used for signal quality detection but it too slow and should be optimized to only neccesary parameters

    def initialize_parameters(self,vis=False):

        rate, sig = wav.read(filename=self.whole_name)
        self.total_length = sig.shape[0]/rate

        if self.total_length < 2.5:
            print ('Recording is too short')

        self.sampling_rate = rate
        self.sample_entropy, normalized_amplitude_envelope, auto_corr_sig_norm = seSQI(sig, rate, self.total_length)
        self.normalized_amplitude_envelope_kurtosis = scipy.stats.kurtosis(normalized_amplitude_envelope)
        self.normalized_amplitude_envelope_variance = np.var(normalized_amplitude_envelope)
        self.max_correlation_peak = mSQI(sig, rate, self.total_length)
        self.spectral_ratio = sdrSQI(normalized_amplitude_envelope, rate)

            #max_time_to_show = 2, fig = plt.figure()
            #ax0 = fig.add_subplot(111)
            #max_time_to_show=2
            #ax0.plot(t[t<max_time_to_show], sig[t<max_time_to_show], label='signal')
            #ax0.plot(t[t<max_time_to_show], amplitude_envelope[t<max_time_to_show], label='envelope')
            #ax0.set_xlabel("time in seconds")
            #ax0.legend()

        #PHase
        #ax1 = fig.add_subplot(212)
        #ax1.plot(t[1:], instantaneous_frequency)
        #ax1.set_xlabel("time in seconds")
        #ax1.set_ylim(0.0, 120.0)
        #corr_idx=int(time_to_correlate*rate)
        #result = np.correlate(normalized_amplitude_envelope[0:corr_idx], normalized_amplitude_envelope, mode='full')
        #result= result[corr_idx:]**2
        #normalized_result=(result-np.mean(result))/np.std(result)



    def get_features(self, w=50,l=50,h=3,max_frames=10,index=0,setting=features_setting):
        '''
        This function get filename for wav file
        it open the file and running over the spectral features using
        0.2 winlen and 0.05 step with 50 height(default
        that means that 2.5second for each frame is analysed
        thefunction using 3types of speech feature from James Lyons 2012 code
        nfilt - the number of filters applied for each winlen in Frame()
        height - the number of analysis window containing all the data
        as winstep=0.05. 20steps is 1 second, thus, 50step equals to 2.5 seconds.

        The function output for each frame the 50x50(default) features using:
        mfcc
        mfcc deltas
        log Mel-filterbank energy
        which creates 3x50x50 feature for each frame.
        the latter, might use to feed cnn for hb classification

        '''

        (rate, sig) = wav.read(filename=self.whole_name)
        nfilt = w
        first_mels = mfcc(signal=sig, samplerate=rate, winlen=0.2, winstep=0.05, numcep=50,
                          nfilt= nfilt, nfft=512, lowfreq=0, highfreq=rate / 4, preemph=0.97, ceplifter=22,
                          appendEnergy=True) #Returns a matrix size nXm [n - # of frames, m - numcep]
        w1, h1 = first_mels.shape

        d_mfcc_feat = delta(first_mels, 2)
        w2, h2 = d_mfcc_feat.shape


        fbank_feat = logfbank(sig, samplerate=rate, winlen=0.2, winstep=0.05,
                              nfilt=w, nfft=512, lowfreq=0, highfreq=rate / 4, preemph=0.97)
        w3, h3 = fbank_feat.shape

        if w1 != w2:
            print("error in dimensions")
            return False

        num_frames = int(len(first_mels) / nfilt)

        if num_frames < max_frames:
            X = np.empty((num_frames, 3, nfilt, nfilt))
        else:
            num_frames = max_frames
            X = np.empty((num_frames, 3, nfilt, nfilt))

        for file_num in range(num_frames):

            X[file_num][0] = first_mels[l * file_num:l + l * file_num]
            X[file_num][1] = d_mfcc_feat[l * file_num:l + l * file_num]
            X[file_num][2] = fbank_feat[l * file_num:l + l * file_num]
        # we will work with first frame only right now
        return X[index]




def datalist(foldername):

    os.chdir(foldername)
    file_number = 0
    print('available csv files:')
    for file in glob.glob("*.csv"):
        print(file)
        file_number+=1

    if file_number==1:
        filename=foldername+'\\'+file
    else:
        input_file_name=input('too few or too many .csv were found, please write the chosen one')
        filename=foldername+'\\'+input_file_name

    print(filename)

    return pd.read_csv(filename, header=-1)



def data_from_file(filename, max_frames=10, nfilt=50, height=50):

    '''
    This function get filename for wav file
    it open the file and running over the spectral features using
     0.2 winlen and 0.05 step with 50 height(default
    that means that 2.5second for each frame is analysis
    the function using 3types of speech feature from James Lyons 2012 code
    nfilt - the number of filters applied for each winlen in Frame()
    height - the number of analysis window containing all the data
    as winstep=0.05. 20steps is 1 second, thus, 50step equals to 2.5 seconds.

    The function output for each frame the 50x50(default) features using:
    mfcc
    mfcc deltas
    log Mel-filterbank energy
    which creates 3x50x50 feature for each frame.
    the later, might use to feed cnn for hb classification

    '''
    (rate, sig) = wav.read(filename=filename)

    first_mels= mfcc(signal=sig, samplerate=rate, winlen=0.2, winstep=0.05, numcep=50,
         nfilt=nfilt, nfft=512, lowfreq=0, highfreq=rate/4, preemph=0.97, ceplifter=22, appendEnergy=True)
    w1, h1 = first_mels.shape

    d_mfcc_feat = delta(first_mels, 2)
    w2, h2 = d_mfcc_feat.shape

    fbank_feat = logfbank(sig,samplerate=rate,winlen=0.2,winstep=0.05,
          nfilt=nfilt,nfft=512,lowfreq=0,highfreq=rate/4,preemph=0.97)
    w3, h3 = fbank_feat.shape


    if w1!= w2 :
        print("error in dimensions")
        return False

    num_frames=int(len(first_mels)/nfilt)
    if num_frames < max_frames:
        X = np.zeros((num_frames, 3, nfilt, nfilt))
    else:
        X = np.zeros((max_frames, 3, nfilt,nfilt))



    for file_num in range(num_frames):


        X[file_num][0] = first_mels[50*file_num:height+height*file_num]
        X[file_num][1] = d_mfcc_feat[50*file_num:height+height*file_num]
        X[file_num][2] = fbank_feat[50*file_num:height+height*file_num]

#we will work with first frame only right now
    return X[0]



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
''''''


