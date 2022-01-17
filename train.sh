# /bin/bash
python -m torch.distributed.run --nproc_per_node=1 main.py -p ./config.yaml