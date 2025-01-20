from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./test_logs")
for i in range(10):
    writer.add_scalar("Test/Loss", i ** 2, i)
writer.close()