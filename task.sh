MODEL=$1
MODE=$2
OPT_PATH=/home/xujiaming/xujiaming/data/Flexgen_data/opt_weights_np
OFFLOAD_DIR=/home/xujiaming/xujiaming/data/Flexgen_data/Offload
export TRANSFORMERS_CACHE=/home/xujiaming/xujiaming/data/Flexgen_data/.cache
export HF_HOME=/home/xujiaming/xujiaming/data/Flexgen_data/.cache/huggingface

case $MODE in 

  inference)

    case $MODEL in

      opt-1.3b)
        CUDA_VISIBLE_DEVICES=0 python3 -m flexgen.flex_opt \
        --model facebook/opt-1.3b \
        --path $OPT_PATH \
        --offload-dir $OFFLOAD_DIR \
        --prompt-len 256 --gen-len 32 --gpu-batch-size 128 --num-gpu-batches 2 \
        ;;

      opt-30b)
        CUDA_VISIBLE_DEVICES=0 python3 -m flexgen.flex_opt \
        --model facebook/opt-30b \
        --path $OPT_PATH  \
        --offload-dir $OFFLOAD_DIR \
        --prompt-len 256 --gen-len 32 --gpu-batch-size 128 --num-gpu-batches 2 --percent 0 100 0 100 100 0 --compress-cache --compress-weight --pin-weight \
        --dfss
        ;;
      opt-6.7b)
        CUDA_VISIBLE_DEVICES=0 python3 -m flexgen.flex_opt \
        --model facebook/opt-6.7b \
        --path $OPT_PATH \
        --offload-dir $OFFLOAD_DIR \
        --prompt-len 1204 --gen-len 32 --gpu-batch-size 1 --num-gpu-batches 1 --percent 100 0 100 0  100 0 
        ;;
    esac

    ;;
  completion)

    case $MODEL in

      opt-6.7b)
        CUDA_VISIBLE_DEVICES=0 python3 -m flexgen.apps.completion \
        --model facebook/opt-6.7b \
        --path $OPT_PATH  \
        --offload-dir $OFFLOAD_DIR
        ;;

      opt-30b)
        CUDA_VISIBLE_DEVICES=0 python3 -m flexgen.apps.completion \
        --model facebook/opt-30b \
        --path $OPT_PATH  \
        --offload-dir $OFFLOAD_DIR \
        --percent 0 100 100 0 100 0
        ;;
      
      opt-max-30b)
        CUDA_VISIBLE_DEVICES=0 python3 -m flexgen.apps.completion \
        --model facebook/opt-30b \
        --path $OPT_PATH  \
        --offload-dir $OFFLOAD_DIR \
        --percent 0 100 100 0 100 0
        ;;
    esac

    ;;
esac
