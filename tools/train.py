import argparse
from dataset.SimpleDataset import RandomDataset
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from model.SimpleNet import Model
import utils.misc as utils
import torch
def train(config):
    dataset = RandomDataset(5, 10)
    sampler = DistributedSampler(dataset)
    rand_loader = DataLoader(dataset=dataset,
                         batch_size=2, shuffle=False, sampler=sampler)
    model = Model(5, 2)
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    local_rank = utils.get_rank()
        
    if torch.cuda.device_count() > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for data, label in rand_loader:
        data = data.cuda()
        label = label.cuda()

        output = model(data)
        label = torch.as_tensor(label, dtype=torch.int64).cuda()
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(model)