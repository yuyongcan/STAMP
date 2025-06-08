### config
DATASET="cifar10c_8_2" # cifar10_c cifar100_c cifar10_c_svhn cifar100_c_svhn cifar10c_8_2 cifar100c_80_20 imagenet_c_noise
METHOD="stamp"       #eata cotta tent rotta sar sotta owttt stamp
GPUS=(5 7) #available gpus
NUM_GPUS=${#GPUS[@]}
NUM_MAX_JOB=$((NUM_GPUS * 2))
i=0


output_path="test-time-val"
#### Useful functions
wait_n(){
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=$NUM_MAX_JOB #num concurrent jobs
  local num_max_jobs=${1:-$default_num_jobs}
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}

test_time_adaptation(){
  ###############################################################
  ###### Run Baselines & NOTE; Evaluation: Target domains  ######
  ###############################################################
  if [ "$METHOD" == "tent" ]; then
    lrs=(0.0001 0.00025 0.001 0.005)
    for lr in ${lrs[*]}; do
      i=$((i + 1))
      wait_n
        CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation-os.py --cfg "cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml" --output_dir "${output_path}/${DATASET}/${METHOD}" \
          --OPTIM_LR "$lr" &
    done
  elif [ "$METHOD" == "cotta" ]; then
    lrs=(0.0001 0.001 0.005)
    rsts=(0.005 0.01 0.02)
    if echo "$DATASET" | grep -q "cifar100"; then
      aps=(0.5 0.72 0.9)
    elif echo "$DATASET" | grep -q "cifar10"; then
      aps=(0.8 0.92 0.95)
    elif echo "$DATASET" | grep -q "imagenet"; then
      aps=(0.05 0.1 0.2)
      rsts=(0.0005 0.001 0.002)
    fi
    for lr in ${lrs[*]}; do
      for rst in ${rsts[*]}; do
        for ap in ${aps[*]}; do
          i=$((i + 1))
          wait_n
            CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation-os.py --cfg "cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml" --output_dir "${output_path}/${DATASET}/${METHOD}" \
              --OPTIM_LR "$lr" --COTTA_RST "$rst" --COTTA_AP "$ap" &
        done
      done
    done

  elif [ "$METHOD" == "eata" ] || [ "$METHOD" == "eataE10" ]; then
    dms=(0.2 0.4 0.6)
    fisher_alphas=(1 50 400 2000)
    lrs=(0.001 0.005 0.01)
    if echo "$DATASET" | grep -q "cifar100"; then
        dms=(0.1 0.2 0.4)
    elif echo "$DATASET" | grep -q "imagenet"; then
        dms=(0.025 0.05 0.1)
        lrs=(0.0001 0.00025 0.0005)
    fi
    for lr in ${lrs[*]}; do
      for dm in ${dms[*]}; do
        for fisher_alpha in ${fisher_alphas[*]}; do
          i=$((i + 1))
          wait_n
            CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation-os.py --cfg "cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml" --output_dir "${output_path}/${DATASET}/${METHOD}" \
              --OPTIM_LR "$lr" --EATA_DM "$dm" --EATA_FISHER_ALPHA "$fisher_alpha" &
        done
      done
    done
  elif [ "$METHOD" == "sar" ]; then
    rsts=(0.1 0.2 0.3)
    lrs=(0.001 0.005 0.01)
    if echo "$DATASET" | grep -q "imagenet"; then
      lrs=(0.0001 0.00025 0.0005)
    fi
    for lr in ${lrs[*]}; do
      for rst in ${rsts[*]}; do
          i=$((i + 1))
          wait_n
          CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation-os.py --cfg "cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml" --output_dir "${output_path}/${DATASET}/${METHOD}" \
              --OPTIM_LR "$lr" --SAR_RESET_CONSTANT "$rst" &
      done
    done
  elif [ "$METHOD" == "rotta" ]; then
    lrs=(0.0001 0.001 0.005)
    for lr in ${lrs[*]}; do
      i=$((i + 1))
      wait_n
        CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation-os.py --cfg "cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml" --output_dir "${output_path}/${DATASET}/${METHOD}" \
          --OPTIM_LR "$lr"&
    done
  elif [ "$METHOD" == "owttt" ]; then
    da_scales=(0.2 1. 5.)
    if echo "$DATASET" | grep -q "cifar100"; then
      lrs=(0.00001 0.0001 0.0005)
    elif echo "$DATASET" | grep -q "cifar10"; then
      lrs=(0.0001 0.001 0.005)
    elif echo "$DATASET" | grep -q "imagenet"; then
      lrs=(0.00001 0.000025 0.0001)
      da_scales=(0.1 0.4 1.)
    fi
    for lr in ${lrs[*]}; do
      for da_scale in ${da_scales[*]}; do
        i=$((i + 1))
        wait_n
          CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation-os.py --cfg "cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml" --output_dir "${output_path}/${DATASET}/${METHOD}" \
            --OPTIM_LR "$lr" --OWTTT_DA_SCALE "$da_scale" &
      done
    done
  elif [ "$METHOD" == "sotta" ]; then
    lrs=(0.0001 0.001 0.005)
    if echo "$DATASET" | grep -q "cifar100"; then
      thresholds=(0.5 0.66 0.8)
    elif echo "$DATASET" | grep -q "cifar10"; then
      thresholds=(0.9 0.99 0.995)
    elif echo "$DATASET" | grep -q "imagenet"; then
      thresholds=(0.2 0.33 0.5)
    fi
    for lr in ${lrs[*]}; do
      for threshold in ${thresholds[*]}; do
        i=$((i + 1))
        wait_n
          CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation-os.py --cfg "cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml" --output_dir "${output_path}/${DATASET}/${METHOD}" \
            --OPTIM_LR "$lr" --SOTTA_THRESHOLD "$threshold" &
      done
    done
  elif [ "$METHOD" == "odin" ]; then
    temps=(1 2 5 10 20 50 100 200 500 1000)
    mags=(0 0.0005 0.001 0.0015 0.002 0.0025 0.003 0.0035 0.004 0.005)
    for temp in ${temps[*]}; do
      for mag in ${mags[*]}; do
        i=$((i + 1))
        wait_n
          CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation-os.py --cfg "cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml" --output_dir "${output_path}/${DATASET}/${METHOD}" \
            --ODIN_TEMP "$temp" --ODIN_MAG "$mag" &
      done
    done
  elif [ "$METHOD" == "stamp" ]; then
    if echo "$DATASET" | grep -q "cifar100_c"; then
      lrs=(0.0001 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.5)
      alphas=(0.3 0.4 0.5 0.6 0.7 0.8 0.9)
    elif echo "$DATASET" | grep -q "cifar10_c"; then
      lrs=(0.00005 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1)
      alphas=(0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6)
    elif echo "$DATASET" | grep -q "cifar10c"; then
      lrs=(0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 1)
      alphas=(0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85)
    elif echo "$DATASET" | grep -q "imagenet"; then
      lrs=(0.0001 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1)
      alphas=(0.2 0.3 0.4 0.5 0.6 0.7 0.8)
    elif echo "$DATASET" | grep -q "cifar100c"; then
      lrs=(0.0001 0.001 0.002 0.005 0.01 0.02 0.05 0.1)
      alphas=(0.3 0.4 0.5 0.6 0.7 0.8 0.9)
    fi
    for lr in ${lrs[*]}; do
      for alpha in ${alphas[*]}; do
          i=$((i + 1))
          wait_n
            CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python test-time-validation-os.py --cfg "cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml" --output_dir "${output_path}/${DATASET}/${METHOD}" \
              --OPTIM_LR "$lr" --STAMP_ALPHA "$alpha" &
        done
      done
    done
  fi
}


test_time_adaptation

