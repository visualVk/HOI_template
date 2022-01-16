from dataset.SimpleDataset import RandomDataset
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from config.config import config
from model.SimpleNet import Model
import torch
def train():
    dataset = RandomDataset(5, 10)
    sampler = DistributedSampler(dataset)
    rand_loader = DataLoader(dataset=dataset,
                         batch_size=2, shuffle=False, sampler=sampler)
    model = Model(5, 2)
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    model = DDP(model, device_ids=config.LOCAL_RANK, output_device=config.LOCAL_RANK)
    
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