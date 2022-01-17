# Code Template of DDP (One Node With Multi Graphics Cards)

## directories

- config  
  buildin configuration
- data  
  train data, val data or checkpoint
- dataset  
  customed dataset
- logs  
  summary writer logs or others
- model  
  Network you want to train
  helper of training network, eg. SMPN implemented BaseModel
- tools  
  train or val script
- utils  
  some utils to help create

**Note: config.yaml in root directory will be used to configure buildin configuration**

## How to use it
```shell
# window
python -m torch.distributed.run --nproc_per_node=1 main.py -p ./config.yaml
# linux
sh train.sh
```