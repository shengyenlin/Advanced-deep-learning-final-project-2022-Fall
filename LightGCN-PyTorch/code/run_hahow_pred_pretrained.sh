ep=2150
layer=5
dim=1024
task='course'

python hahow_pred.py --decay=1e-4 --lr=0.001 --layer=${layer} --seed=7777 --dataset="hahow_${task}" --topks="[10,20,50]" --recdim=${dim} --bpr_batch 500 --testbatch 500 --epochs 5000 \
                     -w ../models/lgn-hahow_${task}-${layer}-${dim}_${ep}ep.pth.tar

ep=20000
layer=2
dim=768
task='course'

python hahow_pred.py --decay=1e-4 --lr=0.001 --layer=${layer} --seed=7777 --dataset="hahow_${task}" --topks="[10,20,50]" --recdim=${dim} --bpr_batch 500 --testbatch 500 --epochs 5000 \
                     -w ../models/lgn-hahow_${task}-${layer}-${dim}_${ep}ep.pth.tar \
                     --cos

ep=250
layer=3
dim=1024
task='group'

python hahow_pred.py --decay=1e-4 --lr=0.001 --layer=${layer} --seed=7777 --dataset="hahow_${task}" --topks="[10,20,50]" --recdim=${dim} --bpr_batch 500 --testbatch 500 --epochs 5000 \
                     -w ../models/lgn-hahow_${task}-${layer}-${dim}_${ep}ep.pth.tar

ep=7000
layer=2
dim=768
task='group'

python hahow_pred.py --decay=1e-4 --lr=0.001 --layer=${layer} --seed=7777 --dataset="hahow_${task}" --topks="[10,20,50]" --recdim=${dim} --bpr_batch 500 --testbatch 500 --epochs 5000 \
                     -w ../models/lgn-hahow_${task}-${layer}-${dim}_${ep}ep.pth.tar \
                     --cos