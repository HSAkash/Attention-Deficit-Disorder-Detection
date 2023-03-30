"""## Import dependencies"""

import mne
import sklearn
import scipy.io
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin,BaseEstimator

# Spliting by group
from sklearn.model_selection import GroupKFold,LeaveOneGroupOut

# Modeling
from tensorflow.keras.layers import Input,Dense,Flatten,GRU,Conv1D, Concatenate, concatenate
from tensorflow.keras.models import Model
import tensorflow as tf


np.random.seed(42)

"""## Download helperfunction"""


from helper_functions import (
    plot_loss_curves,
    create_tensorboard_callback
)

"""## History class"""

class History:
  
  def __init__(self, history=None):
    self.history = {}
    self.create_history_object(history)

  def create_history_object(self, history):
    if history:
      for his in history.history.keys():
        self.history[his] = history.history[his]
  
  def add_history(self, history):
    if not self.history:
      self.create_history_object(history)
      return
    for his in history.history.keys():
      self.history[his] += history.history[his]

"""## Mat data mne

### Mat data mne 60 frequency
"""

mne.set_log_level("WARNING")
def convertmat2mne_60(data):
    ch_names = ['Fz', 'Cz', 'Pz', 'C3', 'T3', 'C4', 'T4', 'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'P3', 'P4', 'T5', 'T6', 'O1', 'O2']
    ch_types = ['eeg'] * 19
    sampling_freq=128
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq, verbose=0)
    info.set_montage('standard_1020', verbose=0)
    data = np.moveaxis(data, 0, -1)
    data=mne.io.RawArray(data, info, verbose=0)
    data.set_eeg_reference(verbose=0)
    epochs=mne.make_fixed_length_epochs(data,duration=4,overlap=0, verbose=0)
    return epochs.get_data()

"""### Mat data mne 1-30 frequency"""

def convertmat2mne_30(data):
    ch_names = ['Fz', 'Cz', 'Pz', 'C3', 'T3', 'C4', 'T4', 'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'P3', 'P4', 'T5', 'T6', 'O1', 'O2']
    ch_types = ['eeg'] * 19
    sampling_freq=128
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq, verbose=0)
    info.set_montage('standard_1020', verbose=0)
    data = np.moveaxis(data, 0, -1)
    data=mne.io.RawArray(data, info, verbose=0)
    data.set_eeg_reference(verbose=0)
    data.filter(l_freq=1,h_freq=30)
    epochs=mne.make_fixed_length_epochs(data,duration=4,overlap=0, verbose=0)
    # return np.moveaxis(epochs.get_data(), 1, -1)
    return epochs.get_data()

def get_mne_dataset_60(list_path):
    dataset = []
    for x in tqdm(list_path):
        data = scipy.io.loadmat(x)[x.split('/')[-1].split('.')[0]]
        data = convertmat2mne_60(data)
        dataset.append(data)
    return dataset

def get_mne_dataset_30(list_path):
    dataset = []
    for x in tqdm(list_path):
        data = scipy.io.loadmat(x)[x.split('/')[-1].split('.')[0]]
        data = convertmat2mne_30(data)
        dataset.append(data)
    return dataset

"""## Get label function"""

labels_dict = {"ADHD":1, "Control":0}
def get_label(filePath):
    return labels_dict[filePath.split("/")[-2]]

"""## Data path"""

data_dir = "ADHD_DATA"

"""## Get all file path"""

filePath_list = glob(f"{data_dir}/*/*.mat")
len(filePath_list)

"""## Shuffle filepath list"""

filePath_list = sklearn.utils.shuffle(filePath_list, random_state=0)

"""## Get labels"""

labels_list = [get_label(x) for x in filePath_list]
len(labels_list)

"""## get data"""

data_list_60 = get_mne_dataset_60(filePath_list)
data_list_30 = get_mne_dataset_30(filePath_list)
np.shape(data_list_60), np.shape(data_list_30)

"""## Get labels """

label_list, groups_list = [], []
for i, data in enumerate(data_list_60):
    label_list.append([labels_list[i]]*len(data))
    groups_list.append([i]*len(data))

"""## List to numpy"""

data_array_60=np.concatenate(data_list_60)
data_array_30=np.concatenate(data_list_30)
label_array=np.concatenate(label_list)
group_array=np.concatenate(groups_list)

data_array_60=np.moveaxis(data_array_60,1,2)
data_array_30=np.moveaxis(data_array_30,1,2)

del(data_list_60)
del(data_list_30)
del(label_list)
del(groups_list)

data_array_60.shape, data_array_30.shape

"""## Scaling"""

class StandardScaler3D(BaseEstimator,TransformerMixin):
    #batch, sequence, channels
    def __init__(self):
        self.scaler = StandardScaler()
        # self.scaler = MinMaxScaler()

    def fit(self,X,y=None):
        self.scaler.fit(X.reshape(-1, X.shape[2]))
        return self

    def transform(self,X):
        return self.scaler.transform(X.reshape( -1,X.shape[2])).reshape(X.shape)

"""## Spliting data by group and scaling"""

gkf=GroupKFold()
for train_index, val_index in gkf.split(data_array_60, label_array, groups=group_array):

    train_features_60,train_labels=data_array_60[train_index],label_array[train_index]
    val_features_60,val_labels=data_array_60[val_index],label_array[val_index]

    scaler_60=StandardScaler3D()
    train_features_60=scaler_60.fit_transform(train_features_60)
    val_features_60=scaler_60.transform(val_features_60)


    train_features_30=data_array_30[train_index]
    val_features_30=data_array_30[val_index]

    scaler_30=StandardScaler3D()
    train_features_30=scaler_30.fit_transform(train_features_30)
    val_features_30=scaler_30.transform(val_features_30)


    break

"""## Create model1"""

tf.random.set_seed(42)

"""### Create  check points"""

checkpoint_path = "heckpoint/cp.ckpt"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    monitor='val_accuracy',
    save_best_only=True
)
tensorboard_dir = "tensorboard"
tensorboard_name = 'ADHD_model'
checkpoint_path

"""### Modeling"""

def block(input):
  conv1 = Conv1D(32, 2, strides=2,activation='relu',padding="same")(input)
  conv2 = Conv1D(32, 4, strides=2,activation='relu',padding="causal")(input)
  conv3 = Conv1D(32, 8, strides=2,activation='relu',padding="causal")(input)
  x = concatenate([conv1,conv2,conv3],axis=-1)
  return x

"""#### Model part1"""

input_60= Input(shape=train_features_60.shape[1:])
block1_60=block(input_60)
block2_60=block(block1_60)
block3_60=block(block2_60)
gru_out1_60 = GRU(32,activation='tanh',return_sequences=True)(block3_60)
gru_out2_60 = GRU(32,activation='tanh',return_sequences=True)(gru_out1_60)
gru_out_60 = concatenate([gru_out1_60,gru_out2_60],axis=-1)
gru_out3_60 = GRU(32,activation='tanh',return_sequences=True)(gru_out_60)
gru_out_60 = concatenate([gru_out1_60,gru_out2_60,gru_out3_60])
gru_out4_60 = GRU(32,activation='tanh')(gru_out_60)
gru_out4_60.shape

"""#### Model part2"""

input_30= Input(shape=train_features_30.shape[1:])
block1_30=block(input_30)
block2_30=block(block1_30)
block3_30=block(block2_30)
gru_out1_30 = GRU(32,activation='tanh',return_sequences=True)(block3_30)
gru_out2_30 = GRU(32,activation='tanh',return_sequences=True)(gru_out1_30)
gru_out_30 = concatenate([gru_out1_30,gru_out2_30],axis=-1)
gru_out3_30 = GRU(32,activation='tanh',return_sequences=True)(gru_out_30)
gru_out_30 = concatenate([gru_out1_30,gru_out2_30,gru_out3_30])
gru_out4_30 = GRU(32,activation='tanh')(gru_out_30)
gru_out4_30.shape

"""#### Combine two model"""

combine = concatenate([gru_out4_60,gru_out4_30])
combine.shape

output = Dense(1,activation='sigmoid')(combine)
model = Model(inputs=[input_60, input_30], outputs=output)

"""#### Compile model"""

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

"""### Summary model"""

model.summary()


"""### Fit the model epoch 10"""

history = model.fit(
    [train_features_60, train_features_30],
    train_labels,
    validation_data = ([val_features_60, val_features_30], val_labels),
    epochs=10,
    batch_size=128,
    callbacks = [
        create_tensorboard_callback(
            dir_name = tensorboard_dir,
            experiment_name = tensorboard_name
        ),
        checkpoint_callback
    ]
)
histories = History(history)


"""### Evaluate

#### Evaluate model
"""

model.evaluate([val_features_60, val_features_30], val_labels)

"""#### Copy Model and load best weight"""

best_model = tf.keras.models.clone_model(model)
best_model.load_weights(checkpoint_path)
best_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

"""#### Evalute best model"""

best_model.evaluate([train_features_60, train_features_30], train_labels), best_model.evaluate([val_features_60, val_features_30], val_labels)


"""### Save best model"""

best_model.save(f"best_model.h5")
