# /bin/bash
clear
# note: you must change cards according to your GPUS
cards=2
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.run --nproc_per_node=$cards ../upt_main_sh.py --backend=nccl
