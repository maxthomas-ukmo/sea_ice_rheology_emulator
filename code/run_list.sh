conda init
conda activate ML_foundation_project

# get current date and time
now=$(date +"%Y_%m_%d_%H_%M_%S")

# make results dir path
results=../results/initial_testing/$1-$now 
mkdir -p $results

if [ "$1" == "E0" ]; then
    python modelling.py --data_points 50000 > $1.log
elif [ "$1" == "E1" ]; then
    python modelling.py --data_points 10000 --model_type randomforestregressor --features siconc,sithic,utau_ai,utau_oi,vtau_ai,vtau_oi,sishea,sistre,sig1_pnorm,sig2_pnorm,sivelu,sivelv --labels sivelv > $1.log
elif [ "$1" == "E2" ]; then
    python modelling.py --data_points 10000 --model_type sgdregressor --features siconc,sithic,utau_ai,utau_oi,vtau_ai,vtau_oi,sishea,sistre,sig1_pnorm,sig2_pnorm,sivelu,sivelv --labels sivelu > $1.log
fi

mv baseline_model.pkl $results
mv hp_search.pkl $results
mv modelling-baseline.png $results
mv modelling-tuned.png $results 

# If a log file exists, move to results directory
if [ -f "$1.log" ]; then
    mv $1.log $results
fi