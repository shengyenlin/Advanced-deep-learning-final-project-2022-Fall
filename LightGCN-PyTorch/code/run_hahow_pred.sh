ep=15000
layer=2
dim=768
#task='course'
task='group'

python hahow_pred.py --decay=1e-4 --lr=0.001 --layer=${layer} --seed=7777 --dataset="hahow_${task}" --topks="[10,20,50]" --recdim=${dim} --bpr_batch 12288 --testbatch 500 --epochs 5000 \
                     -w lgn-hahow_${task}-${layer}-${dim}/lgn-hahow_${task}-${layer}-${dim}_${ep}ep.pth.tar \
                     --cos