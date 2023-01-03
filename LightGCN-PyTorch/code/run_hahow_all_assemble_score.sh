ep=2150
layer=5
dim=1024
task='course'

python hahow_write_assemble_score.py --decay=1e-4 --lr=0.001 --layer=${layer} --seed=7777 --dataset="hahow_${task}" --topks="[10,20,50]" --recdim=${dim} --bpr_batch 12288 --testbatch 500 --epochs 5000 \
                     -w lgn-hahow_${task}-${layer}-${dim}/lgn-hahow_${task}-${layer}-${dim}_${ep}ep.pth.tar \
                     --users seen

ep=20000
layer=2
dim=768
task='course'

python hahow_write_assemble_score.py --decay=1e-4 --lr=0.001 --layer=${layer} --seed=7777 --dataset="hahow_${task}" --topks="[10,20,50]" --recdim=${dim} --bpr_batch 12288 --testbatch 500 --epochs 5000 \
                    -w lgn-hahow_${task}-${layer}-${dim}/lgn-hahow_${task}-${layer}-${dim}_${ep}ep.pth.tar \
                    --users unseen \
                    --cos

ep=250
layer=3
dim=1024
task='group'

python hahow_write_assemble_score.py --decay=1e-4 --lr=0.001 --layer=${layer} --seed=7777 --dataset="hahow_${task}" --topks="[10,20,50]" --recdim=${dim} --bpr_batch 12288 --testbatch 500 --epochs 5000 \
                     -w lgn-hahow_${task}-${layer}-${dim}/lgn-hahow_${task}-${layer}-${dim}_${ep}ep.pth.tar \
                     --users seen

ep=7000
layer=2
dim=768
task='group'

python hahow_write_assemble_score.py --decay=1e-4 --lr=0.001 --layer=${layer} --seed=7777 --dataset="hahow_${task}" --topks="[10,20,50]" --recdim=${dim} --bpr_batch 12288 --testbatch 500 --epochs 5000 \
                     -w lgn-hahow_${task}-${layer}-${dim}/lgn-hahow_${task}-${layer}-${dim}_${ep}ep.pth.tar \
                     --users unseen \
                     --cos

python hahow_merge_assemble_score.py

cp ./assemble/lgn_{course,group}_score.dict.pkl ../../ensemble/data