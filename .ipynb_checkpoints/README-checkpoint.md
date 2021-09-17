# abinferhmmgp


##  Scalable Inference for Hybrid Bayesian Hidden MarkovModel Using Gaussian Process Emission (summmited to JCGS)

We provide the implementation and experiment results for the paper Scalable Inference for Hybrid Bayesian Hidden MarkovModel Using Gaussian Process Emission.


 
## Description

### Methods

* models/hmm_models_v4.py : hmm using gpemission + SVI + AGPE
* models/emission_gp.py : exact gp emission 
* models/emission_gp_rrff.py : approximate gp emission


### Experiments

* exp1_main_section5-1.ipynb, exp2_main_section5-1.ipynb : validation of the weight sampling that reduces the error of ELBO estimator (main-expeiriment section 5.1)
* exp1_appendix.ipynb : validation of the scalable weight sampling (supplementary-experiment section 4.1)
* SM kernel Recovery by SVSS-Ws.ipynb : SM kernel Recovery conducted in [experiment section 5.1](https://arxiv.org/pdf/1910.13565.pdf)


## Requirements

* python >= 3.6
* torch = 1.7
* pandas
* scipy
* scikit-learn version   0.21.2
* scipy                  1.4.1
* 

## Dataset

* datasets/uci_datasets/ : kin8nm and parkinsons set used in our experiment
* datasets/uci_wilson/ : download [UCI Wilson dataset](https://drive.google.com/file/d/0BxWe_IuTnMFcYXhxdUNwRHBKTlU/view) and unzip the downloaded file


## Installation

    git clone https://github.com/ABInferGSM/src.git
    if necessary, install the required module as follows
    pip3 install module-name
    ex) pip3 install numpy 


## Reference 


* http://www.tsc.uc3m.es/~miguel/downloads.php 





