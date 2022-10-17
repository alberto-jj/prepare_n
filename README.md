# prepare_n
The ```prepare_n.py``` function was written by [@alberto-jj](https://github.com/alberto-jj) and [@yjmantilla](https://github.com/yjmantilla).

You can find an example on the function ussage in the ```prepare.ipynb```.

PREPARE(N) is a workflow for preprocessing resting-state electroencephalogram (rs-EEG) signals.

PREPARE(N) stands for (_PREP_)rocess + (_A_)notate non-brain independent components + (_RE_)ject bad epochs + (_N_)ormalize recording-specific variability (optional step).

This fully automated pipeline wraps multiple preprocessing tools, as follows:

 - pyprep
 - mne-icalabel
 - autoreject
 - normalize channel amplitude to control recording-specific variability (optional step)

## pyprep:
Robust average re-referencing, adaptative line-noise correction, and bad channel interpolation were performed using a Python reimplementation of the MATLAB PREP pipeline [1] done by the authors of the PyPREP library [2]. 

The goal of average re-referencing is to get a comparable reference scheme across datasets. Nevertheless, the average reference can be affected by noisy channels.

Thus, the main goal of the PyPREP pipeline is to estimate a robust average reference by excluding these noisy channels from it.



## epochs rejection:
Afterward, length of the epochs is defined and automatic rejection of artifactual epochs is conducted based on the autoreject method [3].


## mne-icalabel:
Of note, a 1 - 100 Hz band-pass Finite Impulse Response (FIR) filter is used prior to Independent component analysis (ICA) to remove low-frequency drifts that would affect the ICA solutions.

ICA artifact correction is carried out using the extended infomax ICA algorithm, available at the MNE library [4]. Code for FastICA method is also available, but the MATLAB original implementation of ICLabel was carried out using extended infomax. 

Then, the Python implementation of ICLabel is used to annotate artifactual and brain components for subsequent automatical rejection (low-pass filter is used following the original ICLabel implementation) [5].

Following mne-icalabel recommendation, this pipeline keeps components from "brain" as well as "others" as we cannot blindly exclude the latter (i.e. neural activity cannot be ruled out in those components annotated as "others"), see mne-icalabel documentation for details [5].

## post ICA epochs rejection:
As recommended by autoreject authors, a second rejection after ICA is recommended to avoid non-typical ocular artifacts (not dropped in ICA)  [3].


## normalization of channel amplitude vector (optional step):
Finally, to model the variability by a recording-specific scaling factor and to “normalize” each recording by dividing its channel data by a recording-specific constant (Huber mean), with a Python implementation of previously published methods [6].

Of note, this last step is not recommended if the user will extract relative features from the rs-EEG (such as relative power, and relative power spectral density).




### References:

[1] Bigdely-Shamlo N, et al. The PREP pipeline: standardized preprocessing for large-scale EEG analysis. Front Neuroinform. 2015; 9: 16.

[2] Appelhoff S, et al. PyPREP: A Python implementation of the preprocessing pipeline (PREP) for EEG data. (0.4.2). Zenodo. 2022. [https://doi.org/10.5281/zenodo.6363576](https://doi.org/10.5281/zenodo.6363576)

[3] Jas M, et al. Autoreject: Automated artifact rejection for MEG and EEG data. Neuroimage. 2017 Oct 1;159:417-429.

[4] Gramfort A, et al. MNE software for processing MEG and EEG data. Neuroimage. 2014 Feb 1;86:446-60. 

[5] Li A, et al. MNE-ICALabel: Automatically annotating ICA components with ICLabel in Python. Journal of Open Source Software. 2022; 7(76): 4484. 

[6] Bigdely-Shamlo N, et al. Automated EEG mega-analysis I: Spectral and amplitude characteristics across studies. Neuroimage. 2020 Feb 15;207:116361.
