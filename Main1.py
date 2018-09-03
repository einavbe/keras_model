from utils import create_full_csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model_functions import datalist,record,plot_confusion_matrix,features_setting,model_setting
from tempfile import TemporaryFile
from sklearn.metrics import confusion_matrix,roc_curve,auc
import matplotlib.pyplot as plt
from scipy import interp
import os,glob
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras_functions import train_model,test_model


def Main():
    '''

    :return:
    '''
#Main Funcion,
   # phase=input('choose phase train/test')
    phase='train'
    np.random.seed(0)  # for reproducibility
    nb_classes=2
    rate=2000
    features_module=features_setting(length=50,width=50, samplerate=rate, winlen=0.2, winstep=0.05, numcep=50,
                          nfilt=50, nfft=512, lowfreq=0, highfreq=rate / 4, preemph=0.97, ceplifter=22,
                          appendEnergy=True)
    #TODO : Finish the model_setting and implement
    #keras_model=model_setting(load_weights=True, model_name='default_model','hb_weights.hdf5',nb_filters,nb_classes,nb_hidden,loss,optimizer,nb_epoch,batch_size,class_fact)
    if (phase=='train'):
        #Load database,usaully saved one,
        #One should modify this function in order to enigineer the features
        norm, abnorm, tot_norm, tot_abnorm = load_train_db(setting=features_module)
        imbalance_factor= int(tot_abnorm / tot_norm)
        #There is one type that bigger
        labels=np.zeros(tot_abnorm+tot_norm)
        #Label for first norm part is 1  and abnorm is 0
        norm=norm[0:tot_norm]
        abnorm=abnorm[0:tot_abnorm]
        labels[tot_norm::]=1
        labels= np_utils.to_categorical(labels,num_classes=nb_classes) # Making this a matrix of classes instead of a single vector
        random_seed=2

        #Split data to train & validation phase by shuffling each of the groups (norm and abnorm)
        #and then concatenating them together

        norm_train, norm_val, label_norm_train, label_norm_val = train_test_split(norm, labels[tot_norm::, :], test_size=0.2, random_seed=random_seed)
        abnorm_train, abnorm_val, label_abnorm_train, label_abnorm_val = train_test_split(abnorm,labels[:tot_abnorm, :], test_size=0.2, random_seed=random_seed)

        X_train=np.concatenate((norm_train,abnorm_train), axis=0)
        Y_train=np.concatenate((label_norm_train,label_abnorm_train), axis=0)
        X_train,Y_train = shuffle(X_train,Y_train,random_state=random_seed)

        X_val= np.concatenate((norm_val,abnorm_val), axis=0)
        Y_val=np.concatenate((label_norm_val,label_abnorm_val), axis=0)
        X_val,Y_val = shuffle(X_val,Y_val,random_state=random_seed)

        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_val.shape)
        print('Y_train shape:', Y_train.shape)
        print('Y_test shape:', Y_val.shape)

        prediction, _ = train_model(X_train,X_val,Y_train,Y_val,model_name='decaymode.json',weights='hb_decay.hdf5')
        visualization(1-prediction,1- Y_val)

def visualization(prediction,Y_test):
    y_test = np.argmax(Y_test, axis=1)
    y_pred = np.argmax(prediction, axis=1)
    #Plot confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    class_names=["Normal","Abnormal"]
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.show()
# Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    nb_classes=2
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], prediction[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), prediction.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= nb_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.figure()
    lw = 2#line width
    plt.plot(fpr[1], tpr[1], color='darkorange',
             lw=lw, label='ROC-1 curve (area = %0.2f)' % roc_auc[1])
    plt.plot(fpr["micro"], tpr["micro"], color='black',
           lw=lw, label='ROC-MicroAvg curve (area = %0.2f)' % roc_auc["micro"])
    plt.plot(fpr["macro"], tpr["macro"], color='green',
           lw=lw, label='ROC-MacroAvg curve (area = %0.2f)' % roc_auc["macro"])

    plt.plot(fpr[0], tpr[0], color='red',
            lw=lw, label='ROC-0 curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def load_train_db(filesfolder=r'C:\Users\Anna-VLS4U\Desktop\PCG RESEARCH\keras_model',setting=features_setting):
    '''
    :param filesfolder:
    :param setting:
    :return:
    '''

    try :
        os.chdir(filesfolder)
        file_number = 0
        print('available npz files:')
        for file in glob.glob("*.npz"):
            print(file)
            file_number += 1

        if file_number == 1:
            filename = filesfolder + '\\' + file
            if input('open this db? y/n  ') == 'y':
                npzfile = np.load(filename)
                norm = npzfile['arr_0']
                abnorm = npzfile['arr_1']
                tot_norm = npzfile['arr_2']
                tot_abnorm = npzfile['arr_3']
            else:
                db_name = input('Enter any db_name.npz to create the db')
                norm, abnorm, tot_norm, tot_abnorm = Model_stats(db_name, filesfolder)

        elif file_number < 1:
            input_file_name = input('Enter any db_name.npz to create the db')
            norm, abnorm, tot_norm, tot_abnorm = Model_stats(input_file_name, filesfolder)

        else:
            input_file_name= input ('Enter the db name required from the apparent list')
            norm, abnorm, tot_norm, tot_abnorm = Model_stats(input_file_name, filesfolder)

    except Exception as inst:
        print(type(inst))  # the exception instance
        print(inst.args) # arguments stored in .args
        print("no db found, creating train db, it might take a while")
         # __str__ allows args to be printed directly
        db_name=input('enter db_name add.npz')
        norm, abnorm, tot_norm, tot_abnorm = create_train_db(db_name=filesfolder+'\\'+db_name,setting=setting)

# at this point we have all the training data loaded in the names: norm abnorm, and we also have the
        #amount of signals in tot_norm and tot_abnorm
    return norm,abnorm,tot_norm,tot_abnorm

def create_train_db(db_name,db_folder=r'C:\Users\Anna-VLS4U\Desktop\PCG RESEARCH\training\training-'
                    ,samp_index=0,setting=features_setting):
    '''

    :param db_name:
    :param db_folder:
    :param samp_index:
    :param setting:
    :return:
    '''
    #specify the folder name
    folders =('a', 'b', 'c', 'd', 'e', 'f')
    normal = np.empty((0, 3, setting.length, setting.width))
    abnormal = np.empty((0, 3, setting.length, setting.width))
    tot_norm = 0
    tot_abnorm = 0
    for folder in folders:
        datafolder = db_folder+folder
        print("working on : ")
        print (datafolder)
        #=r'C:\Users\Anna-VLS4U\Desktop\PCG RESEARCH\training\training-a'
        #The next function get a data list from the csv file in the folder
        #The csv has 2 columns  filename and label
        data_list=datalist(datafolder)
        #here we convert the datalist from Dataframe to matrix
        #data_mtx=pd.DataFrame.as_matrix(data_list)
        for index, row in data_list.iterrows():
            filename = row[0]
            label = row[1]
            crecord = record(name=filename,folder_name=datafolder)


            if samp_index<int(crecord.total_length/setting.total_window_length):
                #TODO : extract data from samp_index!=0 for other parts of code
                features = crecord.get_features(index=samp_index, setting=features_setting)
                if label == 1:
                    normal = np.append(normal, [features], axis=0)
                    tot_norm += 1
                else:
                    abnormal=np.append(abnormal, [features], axis=0)
                    tot_abnorm += 1

    print('saving db')
    np.savez(db_name, normal, abnormal, tot_norm, tot_abnorm)
    print('db saved as:'+db_name)

    return normal, abnormal, tot_norm, tot_abnorm


def Model_stats (db_new_name, filesfolder):
    filename= filesfolder + '\\' + db_new_name
    print('Creating db:' + db_new_name)
    return create_train_db(db_name=filename, setting=features_setting)


if __name__ == "__main__":
    Main()

