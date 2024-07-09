#DATASETS=("cifar10c_8_2" "cifar100c_80_20" "imagenet_c_places" "imagenet_c_text" "cifar10_c_noise" "cifar10_c_svhn" "cifar100_c_lsun" "cifar10_c_tiny" "cifar100_c_noise" "cifar100_c_svhn" "cifar100_c_lsun" "cifar100_c_tiny")
DATASETS=("cifar10_c_svhn" "cifar10_c_lsun" "cifar10_c_tiny" "cifar10_c_noise")

#METHODS=("source" "norm_test" "cotta" "eata" "tent" "sar" "rotta" "owttt" "sotta" "stamp")
METHODS=("stamp")

GPU_id=2
test-baseline(){
  DATASET=$1
  METHOD=$2
  GPU_id=$(($3 % 8))
  output_dir="test-time-evaluation-ECCV/${DATASET}/${METHOD}"
  cfg="cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml"
  if [ "$METHOD" == "tent" ]; then
    if echo "$DATASET" | grep -q "cifar100_c";then
      lr=0.0001
    elif echo "$DATASET" | grep -q "cifar10_c";then
      lr=0.001
    elif echo "$DATASET" | grep -q "imagenet";then
      lr=0.00025
    elif echo "$DATASET" | grep -q "cifar10c_8_2";then
      lr=0.0001
    elif echo "$DATASET" | grep -q "cifar100c_80_20";then
      lr=0.001
    fi
    CUDA_VISIBLE_DEVICES="$GPU_id" python test-time-evaluation.py --cfg cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml --output_dir "${output_dir}" \
      --OPTIM_LR "$lr" &
  elif [ "$METHOD" == "eata" ]; then
    if echo "$DATASET" | grep -q "cifar100_c";then
      lr=0.001
      dm=0.1
      fisher_alpha=1.0
    elif echo "$DATASET" | grep -q "cifar10_c";then
      lr=0.001
      dm=0.2
      fisher_alpha=1.0
    elif echo "$DATASET" | grep -q "imagenet";then
      lr=0.0005
      dm=0.025
      fisher_alpha=1.0
    elif echo "$DATASET" | grep -q "cifar100c_80_20";then
      lr=0.001
      dm=0.2
      fisher_alpha=1.0
    elif echo "$DATASET" | grep -q "cifar10c_8_2";then
      lr=0.001
      dm=0.4
      fisher_alpha=50.0
    fi
    CUDA_VISIBLE_DEVICES="$GPU_id" python test-time-evaluation.py --cfg cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml --output_dir "${output_dir}" \
      --OPTIM_LR "$lr" --EATA_DM "$dm" --EATA_FISHER_ALPHA "$fisher_alpha" &
  elif [ "$METHOD" == "cotta" ]; then
    if echo "$DATASET" | grep -q "cifar100_c";then
      lr=0.001
      ap=0.9
      rst=0.005
    elif echo "$DATASET" | grep -q "cifar10_c";then
      lr=0.001
      ap=0.95
      rst=0.005
    elif echo "$DATASET" | grep -q "imagenet";then
      lr=0.0001
      ap=0.05
      rst=0.0005
    elif echo "$DATASET" | grep -q "cifar100c_80_20";then
      lr=0.001
      ap=0.72
      rst=0.01
    elif echo "$DATASET" | grep -q "cifar10c_8_2";then
      lr=0.0001
      ap=0.95
      rst=0.02
    fi
    CUDA_VISIBLE_DEVICES="$GPU_id" python test-time-evaluation.py --cfg cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml --output_dir "${output_dir}" \
      --OPTIM_LR "$lr" --COTTA_AP "$ap" --COTTA_RST "$rst" &
  elif [ "$METHOD" == "sar" ]; then
    if echo "$DATASET" | grep -q "cifar100_c";then
      lr=0.005
      rst=0.2
    elif echo "$DATASET" | grep -q "cifar10_c";then
      lr=0.005
      rst=0.3
    elif echo "$DATASET" | grep -q "imagenet";then
      lr=0.0005
      rst=0.1
    elif echo "$DATASET" | grep -q "cifar100c_80_20";then
      lr=0.005
      rst=0.1
    elif echo "$DATASET" | grep -q "cifar10c_8_2";then
      lr=0.005
      rst=0.1

    fi
    CUDA_VISIBLE_DEVICES="$GPU_id" python test-time-evaluation.py --cfg cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml --output_dir "${output_dir}" \
      --OPTIM_LR "$lr" --SAR_RESET_CONSTANT "$rst" &
  elif [ "$METHOD" == "rotta" ]; then
    if echo "$DATASET" | grep -q "cifar100_c";then
      lr=0.005
    elif echo "$DATASET" | grep -q "cifar10_c";then
      lr=0.005
    elif echo "$DATASET" | grep -q "imagenet";then
      lr=0.005
    elif echo "$DATASET" | grep -q "cifar100c_80_20";then
      lr=0.005
    elif echo "$DATASET" | grep -q "cifar10c_8_2";then
      lr=0.005
    fi
    CUDA_VISIBLE_DEVICES="$GPU_id" python test-time-evaluation.py --cfg cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml --output_dir "${output_dir}" \
      --OPTIM_LR "$lr" &
  elif [ "$METHOD" == "owttt" ]; then
    if echo "$DATASET" | grep -q "cifar100_c";then
      lr=0.00001
      da_scale=1.0
    elif echo "$DATASET" | grep -q "cifar10_c";then
      lr=0.001
      da_scale=5.
    elif echo "$DATASET" | grep -q "imagenet";then
      lr=0.000025
      da_scale=1.
    elif echo "$DATASET" | grep -q "cifar100c_80_20";then
      lr=0.00001
      da_scale=0.2
    elif echo "$DATASET" | grep -q "cifar10c_8_2";then
      lr=0.001
      da_scale=1.
    fi
    CUDA_VISIBLE_DEVICES="$GPU_id" python test-time-evaluation.py --cfg cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml --output_dir "${output_dir}" \
      --OPTIM_LR "$lr" --OWTTT_DA_SCALE "$da_scale" &
  elif [ "$METHOD" == "sotta" ]; then
    if echo "$DATASET" | grep -q "cifar100_c";then
      lr=0.0001
      threshold=0.5
    elif echo "$DATASET" | grep -q "cifar10_c";then
      lr=0.0001
      threshold=0.9
    elif echo "$DATASET" | grep -q "imagenet";then
      lr=0.0001
      threshold=0.33
    elif echo "$DATASET" | grep -q "cifar100c_80_20";then
      lr=0.001
      threshold=0.5
    elif echo "$DATASET" | grep -q "cifar10c_8_2";then
      lr=0.001
      threshold=0.995
    fi
    CUDA_VISIBLE_DEVICES="$GPU_id" python test-time-evaluation.py --cfg cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml --output_dir "${output_dir}" \
      --OPTIM_LR "$lr" --SOTTA_THRESHOLD "$threshold" &
  elif [ "$METHOD" == "odin" ]; then
    if echo "$DATASET" | grep -q "cifar100_c";then
      temp=200
      mag=0.0005
    elif echo "$DATASET" | grep -q "cifar10_c";then
      temp=5
      mag=0
    fi
    CUDA_VISIBLE_DEVICES="$GPU_id" python test-time-evaluation.py --cfg cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml --output_dir "${output_dir}" \
      --ODIN_TEMP "$temp" --ODIN_MAG "$mag" &
  elif [ "$METHOD" == "stamp" ]; then
    if echo "$DATASET" | grep -q "cifar100_c";then
      lr=0.05
      alpha=0.9
    elif echo "$DATASET" | grep -q "cifar10_c";then
      lr=0.1
      alpha=0.25
    elif echo "$DATASET" | grep -q "imagenet";then
      lr=0.01
      alpha=0.8
    elif echo "$DATASET" | grep -q "cifar100c_80_20";then
      lr=0.1
      alpha=0.6
    elif echo "$DATASET" | grep -q "cifar10c_8_2";then
      lr=0.1
      alpha=0.25
    fi
    CUDA_VISIBLE_DEVICES="$GPU_id" python test-time-evaluation.py --cfg cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml --output_dir "${output_dir}" \
      --OPTIM_LR "$lr" --STAMP_ALPHA "$alpha" &
  elif [ "$METHOD" == "norm_test" ]; then
    CUDA_VISIBLE_DEVICES="$GPU_id" python test-time-evaluation.py --cfg cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml --output_dir "${output_dir}" &
  elif [ "$METHOD" == "source" ]; then
    CUDA_VISIBLE_DEVICES="$GPU_id" python test-time-evaluation.py --cfg cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml --output_dir "${output_dir}" &
  fi
}

for DATASET in "${DATASETS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    test-baseline $DATASET $METHOD $GPU_id
  GPU_id=$((GPU_id + 1))
  done
done