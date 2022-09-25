# Adapted by Alberto Jaramillo-Jimenez from https://github.com/arnodelorme/eeg_pipelines/blob/master/mne/process_mne_template.py

import mne
import os
import sys
import autoreject
import numpy as np
import bids
from mne.datasets.eegbci import standardize
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pyprep.prep_pipeline import PrepPipeline


##Wrapper function - open different eeg formats
def mne_open(filename):
    """
    Function wrapper to read many eeg file types.
    Returns the Raw mne object or the RawEpoch object depending on the case. 
    """

    if '.fif' in filename:
        return mne.io.read_raw(filename, preload=True)
    elif '.cnt' in filename:
        return mne.io.read_raw_cnt(filename, preload=True)
    elif '.vhdr' in filename:
        return mne.io.read_raw_brainvision(filename, preload=True)
    elif '.bdf'  in filename:
        return mne.io.read_raw_bdf(filename, preload=True)
    elif '.edf' in filename:
        return mne.io.read_raw_edf(filename, preload=True)
    else:
        return None


      
      
dataset={
'layout':{'extension':'.bdf', 'session':['hc', 'on'], 'suffix':'eeg', 'return_type':'filename'},
    'ch_names':['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz'],
    'path':"D:/EEGs/PhD_datasets/BIDS/california/"
}

#Path of the BIDS folder
bids_root = "D:/EEGs/PhD_datasets/BIDS/california/"
#Seleccionar solo EEG
datatype = 'eeg'
suffix = 'eeg'

#Tarea
task = 'rest' 

DATASET=dataset #DEFINE DATASET

layoutd = DATASET.get('layout', None)

layout = bids.BIDSLayout(DATASET.get('path', None))
eegs = layout.get(**layoutd)
print(len(eegs), eegs)



eeg_file  = eegs[0]



# -----------------
# General parameters
# -----------------

filename = eeg_file
keep_chans  = dataset['ch_names']
epoch_length = 2 #epoch length in seconds
downsample = 500 #downsampling to 500Hz
line_noise = 60

  

      
# Import, channel standarization
raw = mne_open(filename)


# Remove channels which are not needed
standardize(raw) #standardize ch_names
raw.pick_channels(keep_chans)
ch_names = raw.info["ch_names"]
eeg_index = mne.pick_types(raw.info, eeg=True, eog=False, meg=False)
ch_names_eeg = list(np.asarray(ch_names)[eeg_index])

# Add a montage to the data
montage_kind = "standard_1005"
montage = mne.channels.make_standard_montage(montage_kind)

# Extract some info
sample_rate = raw.info["sfreq"]


# PyPREP
# parameters
prep_params = {
    "ref_chs": ch_names_eeg,
    "reref_chs": ch_names_eeg,
    "line_freqs": np.arange(line_noise, sample_rate / 2, line_noise)
    }
# -----------------
# fit PyPREP
# raw_copy = raw.copy()
prep = PrepPipeline(raw, prep_params, montage)
prep.fit()
raw = prep.raw





# Filter the data
raw.filter(l_freq=1, h_freq=None) # bandpassing 1 Hz

# Extract epochs
epochs = mne.make_fixed_length_epochs(raw, duration = epoch_length, preload=True)
epochs.resample(downsample)

# Automated epoch rejection
ar = autoreject.AutoReject(random_state=11,
                           n_jobs=1, verbose=False)
ar.fit(epochs)
epochs_ar, reject_log = ar.transform(epochs, return_log=True)



# Compute FastICA 
filt_epochs = epochs_ar.copy().filter(l_freq=1.0, h_freq=100.0) # bandpassing 100 Hz (as in the MATLAB implementation of ICLabel)
ica = ICA(
    n_components=15,
    max_iter="auto",
    method="fastica",
    random_state=97)
ica.fit(filt_epochs)

# Annotate using mne-icalabel
ic_labels = label_components(filt_epochs, ica, method="iclabel")
labels = ic_labels["labels"]
exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]] # a conservative approach suggested in mne-icalabel
print(f"Excluding these ICA components: {exclude_idx}")

# ica.apply() changes the Raw object in-place, so let's make a copy first:
reconst_epochs = epochs.copy()
ica.apply(reconst_epochs, exclude=exclude_idx)





# Post ICA automated epoch rejection (suggested by Autoreject authors)
ar = autoreject.AutoReject(random_state=11,
                           n_jobs=1, verbose=False)
ar.fit(reconst_epochs)
epochs_ar, reject_log = ar.transform(reconst_epochs, return_log=True)


# To discuss...
# Normalization of recording-specific variability (optional)
# Export preprocessed data
# USE PYBIDS TO GENERATE THE DERIVATIVES PATH

  