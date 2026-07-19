##  Scalable Inference for Hybrid Bayesian Hidden MarkovModel Using Gaussian Process Emission 

We provide the implementation and experiment results for the paper Scalable Inference for Hybrid Bayesian Hidden MarkovModel Using Gaussian Process Emission.

<p align="center">
    <img src="https://github.com/becre2021/abinferhmmgp/blob/main/img/Grapicalmodel2.PNG" width="500" height="250">
</p>


 
## Description

### Methods

* models/hmm_models_v4.py : approximate inference method for HMMGPSM (SVI + AGPE)
* models/emission_gp.py : exact gp emission 
* models/emission_gp_rrff.py : approximate gp emission


### Examples

* Section4.2-SVI.ipynb : HMMGPSM trained by SVI with the described synthetic dataset
* Section4.3-SVI+AGPE.ipynb : HMMGPSM trained by SVI+AGPE with the described synthetic dataset

### Experiments

* experiments/main_exp1-1.py : Section 4.2 experiment with large T
* experiments/main_exp1-2.py : Section 4.3 experiment with large Nt
* experiments/main_exp2.py : Section 5.1 experiment with large T
* experiments/main_exp3.py : Section 5.2 experiment with large Nt



## Requirements

* python >= 3.6
* torch >= 1.7
* pandas
* scikit-learn  
* scipy        
* munkres


## Dataset

* datasets/synthetic/Q6_Fs200.mat/ : synthetic datset used for Section 4.2 experiment
* datasets/synthetic/Q6_Fs1000.mat/ : synthetic datset used for Section 4.3 experiment
* datasets/real/PigCVP10_Set_Dynamic_Downsample_rate1.mat/ : processed [pigcvp dataset](http://www.timeseriesclassification.com/description.php?Dataset=PigCVP) used for Section 5.1 experiment
* datasets/real/cwru_v1.pickle/ : processed [CWRU dataset](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data) used for Section 5.2 experiment 


## Installation

    git clone https://github.com/becre2021/abinferhmmgp
    if necessary, install the required module as follows
    pip3 install module-name
    ex) pip3 install numpy 


## Reference 

* https://github.com/dillonalaird/pysvihmm
* https://github.com/lindermanlab/ssm
* http://www.tsc.uc3m.es/~miguel/downloads.php


## Citation

If you find this repository useful, please cite the following paper:

```bibtex
@article{jung2022scalable,
  title   = {Scalable Inference for Hybrid Bayesian Hidden Markov Model Using Gaussian Process Emission},
  author  = {Jung, Yohan and Park, Jinkyoo},
  journal = {Journal of Computational and Graphical Statistics},
  volume  = {31},
  number  = {3},
  pages   = {666--683},
  year    = {2022},
  doi     = {10.1080/10618600.2021.2023021},
  url     = {https://doi.org/10.1080/10618600.2021.2023021}
}
```




