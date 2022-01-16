# /bin/bash
python -m torch.distributed.run --nproc_pernode=1 main.py