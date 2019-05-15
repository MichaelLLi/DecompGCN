#!/bin/sh

for d in 0.1 0.2 0.5
    do
    for lr in $(seq 0.001 0.001 0.005)
        do
        for wd in 0.01 0.05 0.0005
            do
            for lt in "V1,E1" "V1,V2" 
                do    
                for i in {1..10}
                    do
                    python train.py --task='citeseer' --num_features=3703 --residual=False --layertype=$lt --normalize=True --training_epochs=1000 --lr=$lr --dropout_p=$d --model=GCN --lrd=100  --hidden_units=16 --n_layers=2 --node_feature=True --weight_decay=$wd >> results/${lt}_${d}_${lr}_${wd}
                    done
                done
            done
        done
    done




