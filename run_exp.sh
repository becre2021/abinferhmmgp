#!/bin/bash


python3 main_exp1-1.py --numexp 5 --iteration 40  --numhidden 4;
python3 main_exp1-1.py --numexp 1 --iteration 40  --numhidden 4 --train VBEM;

python3 main_exp1-1.py --numexp 5 --iteration 40  --numhidden 6;
python3 main_exp1-1.py --numexp 1 --iteration 40  --numhidden 6 --train VBEM;

python3 main_exp1-1.py --numexp 5 --iteration 40  --numhidden 8;
python3 main_exp1-1.py --numexp 1 --iteration 40  --numhidden 8 --train VBEM;

python3 main_exp1-1.py --numexp 5 --iteration 40  --numhidden 10;
python3 main_exp1-1.py --numexp 1 --iteration 40  --numhidden 10 --train VBEM;




# python3 main_exp1-2.py --numexp 5 --iteration 40 --numhidden 4;
# python3 main_exp1-2.py --numexp 5 --iteration 40 --numhidden 6;
# python3 main_exp1-2.py --numexp 5 --iteration 40 --numhidden 8;
# python3 main_exp1-2.py --numexp 5 --iteration 40 --numhidden 10;





# python3 main_exp3.py --numexp 2 --iteration 30 ;
# python3 main_exp3.py --numexp 2 --iteration 30 ;
# python3 main_exp3.py --numexp 2 --iteration 30 ;


# setdata=Concrete
# setQ=4
# numrepexp=3
# lrhyp=.005
# iter=1000
# #echo 'run exp'+str($setdata)
# echo 'run exp3'
# {
# CUDA_VISIBE_DEVICES=0 python3 main3_uci_regression.py --filename $setdata --numQ $setQ --numspt 2 --numbatch 1 --ratesamplespt .05 --lrhyp $lrhyp --iter $iter --numrepexp $numrepexp ;
# CUDA_VISIBE_DEVICES=0 python3 main3_uci_regression.py --filename $setdata --numQ $setQ --numspt 3 --numbatch 1 --ratesamplespt .05 --lrhyp $lrhyp --iter $iter --numrepexp $numrepexp ;
# CUDA_VISIBE_DEVICES=0 python3 main3_uci_regression.py --filename $setdata --numQ $setQ --numspt 5 --numbatch 1 --ratesamplespt .05 --lrhyp $lrhyp --iter $iter --numrepexp $numrepexp ;
# #CUDA_VISIBE_DEVICES=0 python3 main3_uci_regression.py --filename $setdata --numQ $setQ --numspt 5 --numbatch 1 --ratesamplespt .05 --lrhyp .005 --iter 500 --numrepexp $numrepexp ;
# #CUDA_VISIBE_DEVICES=0 python3 main3_uci_regression.py --filename $setdata --numQ $setQ --numspt 7 --numbatch 1 --ratesamplespt .05 --lrhyp .005 --iter 500 --numrepexp $numrepexp ;
# # CUDA_VISIBE_DEVICES=0 python3 main3_uci_regression.py --filename $setdata --numQ $setQ --numspt 10 --numbatch 1 --ratesamplespt .05 --lrhyp .005 --iter 500 --numrepexp $numrepexp ;
# # CUDA_VISIBE_DEVICES=0 python3 main3_uci_regression.py --filename $setdata --numQ $setQ --numspt 15 --numbatch 1 --ratesamplespt .05 --lrhyp .005 --iter 500 --numrepexp $numrepexp ;
# # CUDA_VISIBE_DEVICES=0 python3 main3_uci_regression.py --filename $setdata --numQ $setQ --numspt 20 --numbatch 1 --ratesamplespt .05 --lrhyp .005 --iter 500 --numrepexp $numrepexp ;
# } 
