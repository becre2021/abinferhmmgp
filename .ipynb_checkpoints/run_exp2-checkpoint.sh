#!/bin/bash


# python3 main_exp1-1.py --numexp 5 --iteration 40  --numhidden 4;
# python3 main_exp1-1.py --numexp 1 --iteration 40  --numhidden 4 --train VBEM;

# python3 main_exp1-1.py --numexp 5 --iteration 40  --numhidden 6;
# python3 main_exp1-1.py --numexp 1 --iteration 40  --numhidden 6 --train VBEM;

# python3 main_exp1-1.py --numexp 5 --iteration 40  --numhidden 8;
# python3 main_exp1-1.py --numexp 1 --iteration 40  --numhidden 8 --train VBEM;

# python3 main_exp1-1.py --numexp 5 --iteration 40  --numhidden 10;
# python3 main_exp1-1.py --numexp 1 --iteration 40  --numhidden 10 --train VBEM;




########################
# table 5
########################
# python3 main_exp1-2.py --numexp 5 --iteration 40 --numhidden 4;
# python3 main_exp1-2.py --numexp 5 --iteration 40 --numhidden 6;
# python3 main_exp1-2.py --numexp 5 --iteration 40 --numhidden 7;
# python3 main_exp1-2.py --numexp 5 --iteration 40 --numhidden 8;
# python3 main_exp1-2.py --numexp 5 --iteration 40 --numhidden 9;
# python3 main_exp1-2.py --numexp 5 --iteration 40 --numhidden 10;
#python3 main_exp1-2.py --numexp 5 --iteration 40 --numhidden 11 --savepicklepath './jupyters/result_pickle/scalable_exp1-2/';




########################
# table 4
########################
# python3 main_exp1-2.py --numexp 5 --iteration 40 --numhidden 8 --numQ 3 --savepicklepath './jupyters/result_pickle_scalable_exp1-2/';
# python3 main_exp1-2.py --numexp 5 --iteration 40 --numhidden 8 --numQ 6 --savepicklepath './jupyters/result_pickle/scalable_exp1-2/';

#python3 main_exp1-2.py --numexp 1 --iteration 40 --numhidden 8 --numQ 3 --emission 'gpsm' --savepicklepath './jupyters/result_pickle_scalable_exp1-2/';
#python3 main_exp1-2.py --numexp 1 --iteration 40 --numhidden 8 --numQ 6 --emission 'gpsm' --savepicklepath './jupyters/result_pickle_scalable_exp1-2/';


#for figues
#python3 main_exp1-2.py --numexp 3 --iteration 40 --numhidden 8 --numQ 6 --savepicklepath './jupyters/result_pickle_scalable_exp1-22/';








# python3 main_exp1-2.py --numexp 1 --iteration 40 --numhidden 8 --numQ 6 --emission 'gpsm' --init_iteration 1 --savepicklepath './jupyters/result_pickle_scalable_exp1-2_time/';
# # iter 1, iteration time : 577.739
# # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # iter 1, iteration time : 577.739| train acc : 0.780, train lik : -415871.925,    test acc : 0.760, test lik : -178895.993 |e train acc : 0.780, e train lik : -415871.925 e test acc : 0.760, e test lik : -178895.993 



#python3 main_exp1-2.py --numexp 1 --iteration 40 --numhidden 8 --numQ 3 --init_iteration 1 --savepicklepath './jupyters/result_pickle_scalable_exp1-2_time/';
#python3 main_exp1-2.py --numexp 1 --iteration 50 --numhidden 8 --numQ 6 --init_iteration 1 --savepicklepath './jupyters/result_pickle_scalable_exp1-2_time/';








































#python3 main_exp1-2.py --numexp 1 --iteration 40 --numhidden 8 --numQ 3 --emission 'gpsm' --savepicklepath './jupyters/result_pickle_scalable_exp1-2/';
#python3 main_exp1-2.py --numexp 1 --iteration 40 --numhidden 8 --numQ 3 --savepicklepath './jupyters/result_pickle_scalable_exp1-2/';
#python3 main_exp1-2.py --numexp 5 --iteration 40 --numhidden 8 --numQ 6 --savepicklepath './jupyters/result_pickle/scalable_exp1-2/';



#python3 main_exp1-2.py --numexp 1 --iteration 40 --numhidden 8;
#python3 main_exp1-2.py --numexp 1 --iteration 50 --numhidden 8;
#python3 main_exp1-2.py --numexp 5 --iteration 40 --numhidden 10;





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
