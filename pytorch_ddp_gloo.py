import torch
from torch.utils.data import DataLoader, Dataset, random_split, DistributedSampler
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import torch.nn as nn

# A pytorch wrapper of multiprocessing
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup(rank, world_size):
    """
    Args:
    - rank: unique identifier of each process
    - world_size: total number of processes
    """

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    init_process_group(backend="gloo", rank=rank, world_size=world_size)
    

# Classic trainer
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: torch.device,
        save_every: int,
        rank: int
    ):
        self.gpu_id = gpu_id
        # Modified compared to the GPU version
        self.model = DDP(model.to(gpu_id))
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.validation_steps = [{"val_loss: ": 10000.0, "val_acc": 0}]
        self.rank = rank

    
    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = torch.nn.CrossEntropyLoss()(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch, prof):
        # b_sz = len(next(iter(self.train_data)))
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            
            self._run_batch(source, targets)
            prof.step()
    
    def _accuracy(self, outputs, labels):
        outputs = outputs.to(self.gpu_id)
        labels = labels.to(self.gpu_id)
        out = self.model(outputs)
        loss = nn.functional.cross_entropy(out, labels)
        _, preds = torch.max(out, dim=1)
        acc = (torch.tensor(torch.sum(preds == labels).item() / len(preds)))
        val = {"val_loss: ": loss.item(), "val_acc": acc.item()}
        self.validation_steps.append(val)
        print(val)
    
    def _validation_step(self):
        source, targets = next(iter(self.train_data))
        self._accuracy(source, targets)
        
            
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        torch.save(ckp, "checkpoint.pt")
        # print(f"epoch: {epoch}, gpu_id: {self.gpu_id}, val_loss_acc: {self.validation_steps[-1]}")
    
    def train(self, max_epochs):
        self.model.train()
        # try:
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=5, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/rank_{self.rank}'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for i in range(max_epochs):
                print(f"epoch: {i}, gpu_id: {self.gpu_id}, val_loss_acc: {self.validation_steps[-1]}, rank: {self.rank}")
                self.train_data.sampler.set_epoch(i)
                self._run_epoch(i, prof)
                self._validation_step()
                prof.step()
                if (i % self.save_every) == 0:
                    self._save_checkpoint(i)
       

mnist_dataset = MNIST(root="data/", train=True, download=True, transform=transforms.ToTensor())
train_data, validation_data = random_split(mnist_dataset, [50000, 10000])

shape = train_data.dataset.data.shape
input_size = shape[-1] * shape[-2]
num_classes = len(train_data.dataset.classes)


class MnistModel(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.linear(x)
        




def main(rank: int, world_size: int, total_epochs: int, save_every: int):
    ddp_setup(rank, world_size)

    if torch.cuda.is_available():
        gpu_id = 0
    else: 
        gpu_id = "cpu"
    model = MnistModel(input_size, num_classes)
    train_loader = DataLoader(train_data, batch_size=512, shuffle=False, sampler=DistributedSampler(train_data))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainer = Trainer(model, train_loader, optimizer, gpu_id, save_every, rank)
    trainer.train(max_epochs=total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    total_epochs = 20
    save_every = 5
    world_size = 3
    mp.spawn(main, args=(world_size, total_epochs, save_every), nprocs=world_size)