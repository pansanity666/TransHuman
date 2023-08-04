# training
# CUDA_VISIBLE_DEVICES=3 python train_net.py --cfg_file configs/train_or_eval.yaml virt_data_root data/zju_mocap rasterize_root data/zju_rasterization \
# ratio 0.5 H 1024 W 1024 run_mode train jitter True exp_name nhp_distributed_TransTest resume True gpus "0,"


# ===== distributed training ===== 

# 2cards
# CARD=0,1
# PORT=${PORT:-29521}
# NGPU=2

# 4cards
# CARD=4,5,6,7
# PORT=${PORT:-29510}
# NGPU=4

# 8cards.    
CARD=0,1,2,3,4,5,6,7
PORT=${PORT:-29513}
NGPU=8

# experiment saving name 
EXP_NAME="default"  

CUDA_VISIBLE_DEVICES=$CARD  python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port=$PORT train_net.py --cfg_file configs/train_or_eval.yaml \
    run_mode train  \
    exp_name ${EXP_NAME} \
    resume True \
    gpus "${CARD}," \
    distributed True \
