import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler as DS
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from ddp_tutorial.verbose_elastic_test.toy_model import ToyModel


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def train():
    world_size = torch.cuda.device_count()
    print(f"Running on {torch.cuda.device_count()} GPUs")
    
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = rank % world_size
    torch.cuda.set_device(device)
    print(f"Current device: {rank}.")
    
    checkpoint_dir = 'checkpoints'
    batch_size = 8
    data_loading_workers = 0
    epochs = 3
    
    # Get dataloaders here
    print("Initializing train set...")
    train_dataset = None    # TO DO
    train_sampler = DS(train_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=data_loading_workers, persistent_workers=True, sampler=train_sampler)

    print("Initializing dev set...")
    dev_dataset = None # TO DO
    dev_sampler = DS(dev_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=data_loading_workers, persistent_workers=True, sampler=dev_sampler)
    

    # Create model and move it to GPU with id rank
    model = ToyModel().to(device)
    model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=True)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        dev_dataloader.sampler.set_epoch(epoch)
        train_dataloader.sampler.set_epoch(epoch)
        
        print(f'Evaluating Dev Set Before Epoch {epoch}')
        with torch.no_grad():
            model.eval()  # Turn dropout off
            for i, batch in enumerate(dev_dataloader):
                # PRINT OUT BATCH
                
                print('  - Dev Batch ' + str(i + 1) + '/' + str(len(dev_dataloader)))
                loss = loss_fn(outputs, labels)
        
            # Report dev loss on this epoch, ensuring that
            # logging is only done on rank 0 process
            if rank == 0:
                print('Dev Set Evaluation: ')
                # PRINT LOSS HERE
                
        dist.barrier()
        
        if rank == 0:
            print('Running Training Epoch ' + str(epoch))
        model.train()  # Turn dropout back on
           
        # Iterate over training set
        for i, batch in enumerate(train_dataloader):
            # PRINT OUT BATCH
            
            input = torch.randn(20, 10)
            labels = torch.randn(20, 5).to(device)
        
            # Clear the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input)

            loss = loss_fn(outputs, labels)

            logging.info(f'  - [{rank=}] Batch ' + str(i + 1) + '/' + str(len(train_dataloader)))
                
            # Backward pass
            loss.backward()
            
            # Update the model's parameters
            optimizer.step()
            
        # Avoid redundant saving across processes
        # and report training loss on this epoch
        if rank == 0:
            logging.info(f"{epoch=} / {epochs}")
            logging.info('-' * os.get_terminal_size().columns)
            logging.info(f'Epoch {epoch} Training Set Evaluation: ')
            logging.info('-' * os.get_terminal_size().columns)
            train_loss_evaluator.print_report(args, log_to_wandb=log_to_wandb)
            
            model_path = f"{checkpoint_dir}/epoch_{epoch}_batch_{i}.pt"
            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
