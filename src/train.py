import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.data import VDSRTrainDataset
from utils.eval import psnr
from src.vdsr import VDSR

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
from time import time

# learning parameters
BATCH_SIZE = 64
EPOCHS = 80
lr = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VDSR().to(device)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
criterion = nn.MSELoss()

def train(model, dataloader):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    for idx, data in tqdm(enumerate(dataloader), total=int(len(dataloader))):
        lr_images = data[0].to(device)
        hr_images = data[1].to(device)

        # run model, calculate loss
        optimizer.zero_grad()
        outputs = model(lr_images)
        loss = criterion(outputs, hr_images)

        # update parameters
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.4)
        optimizer.step()

        running_loss += loss.item()
        running_psnr += psnr(hr_images, outputs)
    
    final_loss = running_loss / len(dataloader.dataset)
    final_psnr = running_psnr / len(dataloader)
    return final_loss, final_psnr

train_loss = []
train_psnr = []
start = time()
for scale_factor in [2, 3, 4]:
    print(f'Train Model for Scale Factor x{scale_factor}')
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1} of {EPOCHS} >>>')

        dataset = VDSRTrainDataset(scale_factor=scale_factor)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

        train(model, train_loader)

        # using data augmentation
        dataset = VDSRTrainDataset(scale_factor=scale_factor, rotate=True, flip=True)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

        epoch_loss, epoch_psnr = train(model, train_loader)
        print(f'Training PSNR: {epoch_psnr:.3f}')

        train_loss.append(epoch_loss)
        train_psnr.append(epoch_psnr)

        scheduler.step()
end = time()
print(f'Finished training in {((end - start) / 60):.3f} minutes')

matplotlib.style.use('ggplot')

# display statistics
# - loss plot
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../outputs/loss.png')
plt.show()
# - psnr plot
plt.figure(figsize=(10, 7))
plt.plot(train_psnr, color='green', label='train PSNR dB')
plt.xlabel('Epochs')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.savefig('../outputs/psnr.png')
plt.show()

# save the model to the disk
print('Saving model ...')
torch.save(model.state_dict(), f'../outputs/model_epoch_{EPOCHS}.pth')