python hahow_pred.py --decay=1e-4 --lr=0.001 --layer=2 --seed=7777 --dataset="hahow_course" --topks="[10,20,50]" --recdim=1024 --bpr_batch 12288 --testbatch 500 --epochs 5000 \
                    -w lgn-hahow_course-2-1024/lgn-hahow_course-2-1024_3950ep.pth.tar
#python hahow_pred.py --decay=1e-4 --lr=0.001 --layer=2 --seed=7777 --dataset="hahow_group" --topks="[10,20,50]" --recdim=1024 --bpr_batch 12288 --testbatch 500 --epochs 5000 \
#                    -w lgn-hahow_group-2-1024/lgn-hahow_group-2-1024_3950ep.pth.tar