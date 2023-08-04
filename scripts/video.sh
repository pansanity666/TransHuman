
CARD=$1
EPOCH=$2
EXP_NAME="$3"

test_input_view="0,7,15" 
test_target_view="3," # Dummy target view, not used. 
test_mode='model_o_motion_x' # [model_o_motion_o | model_o_motion_x | model_x_motion_x ] 

CMD="python run.py --type visualize --cfg_file configs/performance.yaml  \
test.input_view ${test_input_view} \
test.target_view ${test_target_view} \
run_mode test test.mode ${test_mode} exp_name ${EXP_NAME} \
test.epoch ${EPOCH} test.exp_folder_name performance_${test_mode} gpus "${CARD}," \
test.full_eval True  \
" 

LOG_DIR="./data/result/transhuman/${EXP_NAME}"
                
if [ ! -d $LOG_DIR ]; then
  echo "Creating directory: $LOG_DIR"
  mkdir -p $LOG_DIR

  echo "Saving to: $LOG_DIR"
fi

# CUDA_VISIBLE_DEVICES=${CARD}, nohup $CMD > $LOG_DIR/log_E${EPOCH}_${test_mode}.txt 2>&1 &
CUDA_VISIBLE_DEVICES=${CARD}, $CMD 


