#python main.py --decay=1e-4 --lr=0.001 --layer=2 --seed=7777 --dataset="suspether" --topks="[10,20,50]" --recdim=64 --path "./my_ckpts"

#python main.py --decay=1e-4 --lr=0.001 --layer=2 --seed=7777 --dataset="hahow_group" --topks="[10,20,50]" --recdim=128
python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=7777 --dataset="hahow_group" --topks="[10,20,50]" --recdim=1024 --bpr_batch 12288 --testbatch 500 --epochs 300