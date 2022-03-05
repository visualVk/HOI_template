# /bin/bash
clear
# note: you must change cards according to your GPUS
cards=2
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.run --nproc_per_node=$cards main.py --backend=gloo -p ./config.yaml