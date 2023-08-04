CARD=$1
EPOCH=$2
EXP_NAME="$3"

### training set fitting 
# test_input_view="0,7,15"  
# test_target_view="3,5,10,12,18,20" 
# test_mode='model_o_motion_o'

### pose generalization 
# test_input_view="0,7,15"  
# test_target_view="3,5,10,12,18,20" 
# test_mode='model_o_motion_x'

### identity generalization 
test_input_view="0,7,15"  
test_target_view="3,5,10,12,18,20" 
test_mode='model_x_motion_x' 

### one-shot generalization 
# test_input_view="0,"  
# test_target_view="3,5,10,12,18,20" 
# test_mode='model_x_motion_x' 

CMD="python run.py --type evaluate --cfg_file configs/train_or_eval.yaml gpus "${CARD},"  \
run_mode test test.input_view ${test_input_view} test.target_view ${test_target_view} test.mode ${test_mode} \
exp_name ${EXP_NAME}  test.epoch ${EPOCH} test.exp_folder_name zjumocap_${test_mode}  \
test.full_eval False \
" 

LOG_DIR="./data/result/transhuman/${EXP_NAME}"
                
if [ ! -d $LOG_DIR ]; then
  echo "Creating directory: $LOG_DIR"
  mkdir -p $LOG_DIR

  echo "Saving to: $LOG_DIR"
fi

# CUDA_VISIBLE_DEVICES=${CARD}, nohup $CMD > $LOG_DIR/log_E${EPOCH}_${test_mode}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=${CARD}, $CMD 
