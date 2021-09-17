# abinferhmmgp


##  Scalable Inference for Hybrid Bayesian Hidden MarkovModel Using Gaussian Process Emission (summmited to JCGS)

We provide the implementation and experiment results for the paper Scalable Inference for Hybrid Bayesian Hidden MarkovModel Using Gaussian Process Emission.

![ScreenShot](https://github.com/{becre2021}/{abinferhmmgp}/{main}/{imag/Grapicalmodel2.PNG})

 
## Description

### Methods

* models/hmm_models_v4.py : approximate inference method for HMMGPSM (SVI + AGPE)
* models/emission_gp.py : exact gp emission 
* models/emission_gp_rrff.py : approximate gp emission


### Experiments

* Section4.2-SVI.ipynb : HMMGPSM trained by SVI with the described synthetic dataset
* Section4.3-SVI+AGPE.ipynb : HMMGPSM trained by SVI+AGPE with the described synthetic dataset

* main_exp1-1.py : Section 4.2 experiment with large $T$
* main_exp1-2.py : Section 4.3 experiment with large $N_t$
* main_exp2.py : Section 5.1 experiment with large $T$
* main_exp3.py : Section 5.2 experiment with large $N_t$



## Requirements

* python >= 3.6
* torch = 1.7
* pandas
* scipy
* scikit-learn version   0.21.2
* scipy                  1.4.1
* 

## Dataset

* datasets/synthetic/Q6_Fs200.mat/ : synthetic datset used for Section 4.2 experiment
* datasets/synthetic/Q6_Fs1000.mat/ : synthetic datset used for Section 4.3 experiment
* datasets/real/PigCVP10_Set_Dynamic_Downsample_rate1.mat/ : processed [pigcvp dataset](http://www.timeseriesclassification.com/description.php?Dataset=PigCVP) used for Section 5.1 experiment
* datasets/real/cwru_v1.pickle/ : processed [CWRU dataset](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data) used for Section 5.2 experiment 


## Installation

    git clone https://github.com/ABInferGSM/src.git
    if necessary, install the required module as follows
    pip3 install module-name
    ex) pip3 install numpy 


## Reference 

* https://github.com/dillonalaird/pysvihmm
* https://github.com/lindermanlab/ssm
* https://github.com/becre2021/abinfergsm





