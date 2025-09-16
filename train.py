import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.data = torch.randn(size, 10)
        self.labels = torch.randn(size, 1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def setup():
    dist.init_process_group("nccl")
    print(f"Process {dist.get_rank()} successfully initialized.")


def cleanup():
    current_rank = dist.get_rank()
    dist.destroy_process_group()
    print(f"[Rank {current_rank}] Cleanup complete.")


def run_training(rank, local_rank, world_size, batch_size):
    setup()
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    device = torch.device("cuda", device_id)
    print(f"Process {rank} (local_rank {local_rank}) is using device {device}")
    model = SimpleModel().to(device)
    model = DistributedDataParallel(model, device_ids=[device_id])
    dataset = SimpleDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 10
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        total_loss = 0
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"\nRank {rank} Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(dataloader):.4f}")
    cleanup()


if __name__ == "__main__":
    batch_size = 8
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    run_training(rank, local_rank, world_size, batch_size)
