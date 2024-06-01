**Introduction**

Image classification project using the HAM10K dataset. The HAM10K dataset is imbalanced and contains 7 different disease classes.

Convolutional Neural Network (CNN) architectures have been used to obtain the most efficient set of hyperparameters.

Data augmentation has been utilized.

 **Project stages:**

1. Visualize the data
2. Apply ROS (Random Over Sampling)
3. Create the dataset class
4. Create the CNN architecture with ImageNet pretraining
5. Train and validate the model



**Installing Necessary Modules**




```

import torch
import torchvision
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

import pandas as pd
from imblearn.over_sampling import RandomOverSampler as ROS
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import random
import torch.nn.functional as F
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

RANDOM_SEED = 5

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed : {seed} ")

set_seed(RANDOM_SEED)

```


**Hyper Parameters**

```
lr = 0.001
bs = 256
EPOCHS = 100
FT_EPOCHS = 5
LR_MIN = 1e-6

highest_acc = 0.80

SAVE_CHECKPOINT = True
SAVE_HIGHEST = True

```

1. Data Preprocessing

```

data = pd.read_csv('/kaggle/input/skin-cancer-mnist-ham10000/hmnist_28_28_RGB.csv')
data.head()

# The dataset contains images classified into the following categories:
# - Melanoma (mel)
# - Basal cell carcinoma (bcc)
# - Actinic keratosis and intraepithelial carcinoma/Bowen's disease (akiec)
# - Benign keratosis-like lesions (solar lentigo/seborrheic keratosis and lichen planus-like keratosis, bkl)
# - Dermatofibroma (df)
# - Melanocytic nevus (nv)
# - Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas, and hemorrhage, vasc)

classes = {
    0: ('akiec', 'Actinic Keratosis'),
    1: ('bcc', 'Basal Cell Carcinoma'),
    2: ('bkl', 'Benign Keratosis-Like Lesions'),
    3: ('df', 'Dermatofibroma'),
    4: ('nv', 'Melanocytic Nevus'),
    5: ('vasc', 'Vascular Lesions'),
    6: ('mel', 'Melanoma'),
}
CLASSES = [classes[idx][0] for idx in range(len(classes))]
CLASSES_FULL = [classes[idx][1] for idx in range(len(classes))]
CLASSES, CLASSES_FULL

```

**output**

(['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel'],
 ['Aktinik Keratoz',
  ' Bazal Hücreli Karsinom',
  'Benign Keratoz Benzeri Lezyonlar',
  'Dermatofibroma',
  'Melanositik Nevüs',
  'Vasküler Lezyonlar',
  'Melanom'])

```

sample_images = []
N = len(CLASSES)
for class_ in classes.keys():
    image_idxs = data.label==class_
    N_ = len(data[image_idxs])
    chosen = random.sample(list(np.arange(N_)), k= N)
    images = np.asarray(data[image_idxs].iloc[chosen,:-1])
    
    for img in images:
        sample_images.append(torch.tensor(img.reshape(28,28,3)).permute(2,0,1))
        
s = torch.stack(sample_images)
grid = torchvision.utils.make_grid(s, nrow=N, ncol=N)

plt.figure(figsize=(8,8), dpi=(128))
plt.imshow(grid.permute(1,2,0))
plt.xticks(np.linspace(14,grid.shape[2]-14,7), labels=[f'ornek {idx+1}' for idx in range(N)])
plt.yticks(np.linspace(14,grid.shape[1]-14,7), labels=[f'[{idx}] {cls_}' for idx, cls_ in enumerate(CLASSES_FULL)])
plt.title('HAM10000 Dataset Examples')
plt.legend(CLASSES_FULL)
plt.show(block='off')

```

![image](https://github.com/UserSemanur/Final-Project/assets/108471885/76df65ae-572c-4222-b74e-1434f86d5eeb)



```

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Create subplots to visualize data distribution
fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=96, sharex=True, sharey=True)

# Separate features and labels
x = data.drop(labels='label', axis=1)
y = data.label

# Plot the original class distribution
sns.countplot(x=data['label'], ax=ax[0])
ax[0].set(xlabel='Classes', ylabel='Counts')
ax[0].title.set_text('HAM10000 Disease Counts')

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Copy training data for visualization
x_train_ = x_train.copy()
x_train_.insert(0, 'label', value=y_train.values)
train_df_ = x_train_

# Plot the class distribution in the training set
sns.countplot(x=train_df_['label'], ax=ax[1])
ax[1].title.set_text('Training Set Class Counts')

# ROS (Random Over Sampling) could be applied here
# from imblearn.over_sampling import RandomOverSampler
# oversampler = RandomOverSampler()
# x_train, y_train = oversampler.fit_resample(x_train, y_train)

# Insert labels back into the training and testing sets for visualization
x_train.insert(0, 'label', value=y_train.values)
train_df = x_train

x_test.insert(0, 'label', value=y_test.values)
test_df = x_test

# Plot the class distribution after ROS
sns.countplot(x=train_df['label'], ax=ax[2])
ax[2].title.set_text('After ROS')

# Display the plots
plt.show()

```
**output**

![image](https://github.com/UserSemanur/Final-Project/assets/108471885/5a1bebf5-c35c-4286-a652-6e90d1dc8192)



**3. Creating the Dataset**

```
class HAM10KDS(Dataset):
    def __init__(self, df, transforms=None, selective=True, OR=4, normalized=False, standardized=True):
        self.data = df
        self.y = self.data.label
        self.x = self.data.drop(labels='label', axis=1)
        self.x = np.asarray(self.x).reshape(-1, 28, 28, 3)
        
        # Calculate mean and standard deviation
        self.mean = torch.tensor(np.mean(self.x))
        self.std = torch.tensor(np.std(self.x))
        
        # Convert image data to tensors
        self.x = torch.tensor(self.x, dtype=torch.float32).permute(0, 3, 1, 2)
        
        # Apply Standardization or Normalization to the data
        if standardized:
            self.x = (self.x - self.mean) / self.std
            print('Standardization applied')
        elif normalized:
            self.x = (self.x - torch.min(self.x)) / (torch.max(self.x) - torch.min(self.x))
            print('Normalization applied')

        self.resize = T.Resize((28 * 4, 28 * 4))
        self.OR = OR
        self.tf = transforms
        self.selective = selective

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        label = torch.tensor(self.y.iloc[idx])

        img = self.x[idx]
        img = self.resize(img)

        if self.tf is not None:
            if self.selective:
                if label.item() != self.OR: # Apply transform to classes other than class 4
                    img = self.tf(img)
            else:
                img = self.tf(img)

        return img, label


```
**4. Creating the CNN Model**

Models are obtained and trained from the PyTorch API, and a classification head containing 3 linear layers is added.

```
class CNN(nn.Module):
    
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        
        self.num_classes = num_classes
        print(f'{self.num_classes} adet sınıf var.')
        
        self.model = models.efficientnet_b0(pretrained = False)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            
            nn.Linear(1280, 256, bias=False), 
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, self.num_classes, bias=False),
            nn.BatchNorm1d(self.num_classes),
        )
        self.model.classifier=self.classifier
        
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Model {params} adet parametreye sahiptir.')

    def forward(self, x):
        
        return self.model(x)


```
**5. Training and Validation**


```

# Image augmentation is applied only to the training data
tf = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomAffine(degrees=15, 
                   translate=(0.1, 0.1), 
                   scale=(0.9,1.0), 
                   shear=(10))
])

# Dataset and DataLoader
train_ds = HAM10KDS(train_df, transforms=tf, selective=False)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=True, num_workers=2)

test_ds = HAM10KDS(test_df)
test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, pin_memory=True, num_workers=2)

print('Training data:', len(train_ds),' samples, test data:', len(test_ds), ' samples.')

model = CNN(len(CLASSES)).to(DEVICE)

# Freeze model parameters for later
for p in model.parameters():
    p.requires_grad=False

# Classification head parameters are unfrozen
for p in model.classifier.parameters():
    p.requires_grad=True
# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1) # You can choose from AdamW, Adadelta, Adagrad, Adam, Adamax, ASGD, LBFGS, NAdam, RMSprop, SGD

# Loss function
criterion = nn.CrossEntropyLoss()
eval_criterion = nn.CrossEntropyLoss()

# Learning Rate Scheduler (Dynamic Learning Rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',      # It's set to work in 'min' mode, meaning a metric being minimized is used (e.g., loss).
    factor=0.1,       # Factor by which the learning rate will be reduced. For example, new_lr = old_lr * factor.
    patience=3,       # Patience is used to wait for the performance to improve. If the performance does not improve for the specified 'patience' epochs, the learning rate is reduced.
    threshold=0.0001, # If used in 'rel' mode, this determines how much smaller a new minimum value should be.
    threshold_mode='rel',  # If 'rel', threshold is considered as a ratio. If 'abs', threshold is considered as an absolute value.
    cooldown=0,       # Cooldown period is used to wait for another performance improvement after reducing the learning rate.
    min_lr=0,         # The smallest value the learning rate can drop to.
    eps=1e-08,        # A small number used in operations to prevent undesirable outcomes when dealing with fractional numbers.
    verbose=True      # When set to True, produces console output about reduction operations.
)

```

Standardization applied
Standardization applied
Training data consists of 8012 samples, and test data consists of 2003 samples.
There are 7 classes.
The model has 4369674 parameters.

```
train_acc = []
train_losses = []

test_acc = []
test_losses = []

lrs = []

# Warm Up
lr0 = lr*0.01
lr_step = (lr-lr0)/(len(train_dl)-1)

best_acc = 0.90 * highest_acc

for epoch in range(EPOCHS):
    loader = tqdm(train_dl)
    losses = []  # Average losses per epoch
    accs = []  # Average accuracies per epoch
    correct = 0  # Count of correct predictions
    count = 0  # Count of examples
    
    if epoch > 0:
        lrs += [optimizer.param_groups[0]['lr']] * len(train_dl)  # To track LR
    
    if epoch == FT_EPOCHS:
        for p in model.parameters():
            p.requires_grad = True
        print('End-to-end training starts')

    model.train()
    for bidx, (images, labels) in enumerate(loader):
# Warm up
#             if epoch==0:
#                 lr_ = lr0 + lr_step*bidx

#                 for op in optimizer.param_groups:
#                     op['lr'] = lr_
#                     print(op["lr"])

#                 lrs.append(optimizer.param_groups[0]['lr']) # track lr

            images = images.to(DEVICE)  # Move to GPU
            labels = labels.to(DEVICE)
                
            score = model(images)
            loss = criterion(score, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                pred = torch.argmax(score, -1).detach()
                correct += (pred == labels).sum()  # Count correct predictions
                count += len(labels)
                acc = correct / count  # Overall accuracy
                
                losses.append(loss.item())
                accs.append(acc)
                
                loader.set_description(f'TRAIN | epoch {epoch+1}/{EPOCHS} | acc {acc:.4f} | loss {loss.item():.4f}')
train_acc.append(acc)
train_losses.append(torch.tensor(losses).mean().item())

model.eval()
with torch.no_grad():
    loader = tqdm(test_dl)
    
    losses = []  # Track losses per batch
    accs = []  # Track accuracy per epoch
    
    correct = 0  # Count of correct predictions for the epoch
    count = 0  # Total number of images used for the epoch
    
    for bidx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)  # Move to GPU
        labels = labels.to(DEVICE)

        score = model(images)
        loss = eval_criterion(score, labels)

        pred = torch.argmax(score, -1).detach()
        correct += (pred == labels).sum()
        count += len(labels)

        acc = correct / count
        loader.set_description(f'TEST | epoch {epoch+1}/{EPOCHS} | acc {acc:.4f} | loss {loss.item():.4f}')

        losses.append(loss.item())
        accs.append(acc)
  for op in optimizer.param_groups:
            print(op["lr"])
        
        test_acc.append(acc)
        test_losses.append(torch.tensor(losses).mean().item())
        
        scheduler.step(torch.tensor(acc)) # lr scheduler. LR'yi otomatik olarak ayarlamak için.
        
        if SAVE_CHECKPOINT==True:
            if test_acc[-1]>best_acc:
                best_acc = test_acc[-1].item()

                checkpoint = {
                    'model': model,
                    'losses': [train_losses, test_losses],
                    'accs': [train_acc, test_acc],
                    'lrs': lrs,
                    'best_acc': best_acc,
                    'last_epoch_trained': epoch,
                }
                
                if best_acc > highest_acc and SAVE_HIGHEST==True:
                    old_highest_acc = highest_acc
                    highest_acc = best_acc
                    torch.save(highest_acc, 'highest_acc.pt')
                    print(f'En yüksek acc ({old_highest_acc}) ile aşıldı: {highest_acc}')
                    torch.save(checkpoint, f'{highest_acc:.4f} checkpoint.pt')

                torch.save(checkpoint, 'checkpoint.pt')
                print(f'{best_acc:.4f} acc ile model kaydedildi')
            
        if optimizer.param_groups[0]['lr']<LR_MIN:
            print(f'EARLY STOPPING! LR below {LR_MIN}')
            break



```
**output**

TRAIN | epoch 1/100 | acc 0.4312 | loss 1.7563: 100%|██████████| 32/32 [00:09<00:00,  3.50it/s]

TEST | epoch 1/100 | acc 0.6680 | loss 1.1481: 100%|██████████| 8/8 [00:00<00:00, 10.27it/s]

0.006

TRAIN | epoch 2/100 | acc 0.5619 | loss 1.5251: 100%|██████████| 32/32 [00:03<00:00, 10.19it/s]

TEST | epoch 2/100 | acc 0.6680 | loss 1.1661: 100%|██████████| 8/8 [00:00<00:00,  9.55it/s]

0.006

TRAIN | epoch 3/100 | acc 0.6057 | loss 1.4210: 100%|██████████| 32/32 [00:03<00:00, 10.45it/s]

TEST | epoch 3/100 | acc 0.4723 | loss 1.6892: 100%|██████████| 8/8 [00:00<00:00, 10.26it/s]

0.006

TRAIN | epoch 4/100 | acc 0.6227 | loss 1.3073: 100%|██████████| 32/32 [00:03<00:00, 10.51it/s]

TEST | epoch 4/100 | acc 0.6605 | loss 1.0834: 100%|██████████| 8/8 [00:00<00:00, 10.28it/s]

0.006

TRAIN | epoch 5/100 | acc 0.6482 | loss 1.1435: 100%|██████████| 32/32 [00:03<00:00,  9.33it/s]

TEST | epoch 5/100 | acc 0.6740 | loss 1.0176: 100%|██████████| 8/8 [00:00<00:00,  9.33it/s]

0.006

  0%|          | 0/32 [00:00<?, ?it/s]
  
TRAIN | epoch 6/100 | acc 0.6385 | loss 1.1197: 100%|██████████| 32/32 [00:09<00:00,  3.37it/s]

TEST | epoch 6/100 | acc 0.3080 | loss 2.4727: 100%|██████████| 8/8 [00:00<00:00,  9.97it/s]

0.006

TRAIN | epoch 7/100 | acc 0.6735 | loss 0.9554: 100%|██████████| 32/32 [00:09<00:00,  3.49it/s]

TEST | epoch 7/100 | acc 0.6420 | loss 0.9350: 100%|██████████| 8/8 [00:00<00:00,  9.80it/s]

0.006

TRAIN | epoch 8/100 | acc 0.6920 | loss 0.8597: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 8/100 | acc 0.6181 | loss 1.1944: 100%|██████████| 8/8 [00:00<00:00,  9.86it/s]

0.006

TRAIN | epoch 9/100 | acc 0.7077 | loss 0.7870: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 9/100 | acc 0.6236 | loss 1.0030: 100%|██████████| 8/8 [00:00<00:00,  9.18it/s]

0.006

TRAIN | epoch 10/100 | acc 0.7179 | loss 0.8991: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 10/100 | acc 0.6955 | loss 0.8151: 100%|██████████| 8/8 [00:00<00:00,  9.42it/s]

0.006

TRAIN | epoch 11/100 | acc 0.7249 | loss 0.9348: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 11/100 | acc 0.7139 | loss 0.7912: 100%|██████████| 8/8 [00:01<00:00,  7.74it/s]

0.006

TRAIN | epoch 12/100 | acc 0.7354 | loss 0.7111: 100%|██████████| 32/32 [00:09<00:00,  3.47it/s]

TEST | epoch 12/100 | acc 0.6950 | loss 0.8571: 100%|██████████| 8/8 [00:00<00:00,  9.46it/s]

0.006

TRAIN | epoch 13/100 | acc 0.7443 | loss 0.8333: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 13/100 | acc 0.7129 | loss 0.6885: 100%|██████████| 8/8 [00:00<00:00,  9.64it/s]

0.006

TRAIN | epoch 14/100 | acc 0.7461 | loss 0.7626: 100%|██████████| 32/32 [00:09<00:00,  3.49it/s]

TEST | epoch 14/100 | acc 0.7379 | loss 0.7252: 100%|██████████| 8/8 [00:00<00:00,  8.47it/s]

0.006

The model was saved with an accuracy of 0.7379.

TRAIN | epoch 15/100 | acc 0.7595 | loss 0.8473: 100%|██████████| 32/32 [00:09<00:00,  3.46it/s]

TEST | epoch 15/100 | acc 0.7184 | loss 0.7697: 100%|██████████| 8/8 [00:00<00:00,  8.74it/s]

0.006

TRAIN | epoch 16/100 | acc 0.7595 | loss 0.6964: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 16/100 | acc 0.7219 | loss 0.8016: 100%|██████████| 8/8 [00:00<00:00,  9.39it/s]

0.006

TRAIN | epoch 17/100 | acc 0.7610 | loss 0.5389: 100%|██████████| 32/32 [00:09<00:00,  3.47it/s]

TEST | epoch 17/100 | acc 0.7134 | loss 0.6855: 100%|██████████| 8/8 [00:00<00:00,  9.47it/s]

0.006

TRAIN | epoch 18/100 | acc 0.7770 | loss 0.7861: 100%|██████████| 32/32 [00:09<00:00,  3.47it/s]

TEST | epoch 18/100 | acc 0.7074 | loss 0.7914: 100%|██████████| 8/8 [00:00<00:00,  9.14it/s]

0.006

TRAIN | epoch 19/100 | acc 0.7736 | loss 0.5634: 100%|██████████| 32/32 [00:09<00:00,  3.47it/s]

TEST | epoch 19/100 | acc 0.7379 | loss 0.7211: 100%|██████████| 8/8 [00:00<00:00,  9.10it/s]

0.006

TRAIN | epoch 20/100 | acc 0.7841 | loss 0.7288: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 20/100 | acc 0.7384 | loss 0.7163: 100%|██████████| 8/8 [00:00<00:00,  9.19it/s]

0.006

0.7384 acc ile model kaydedildi

TRAIN | epoch 21/100 | acc 0.7888 | loss 0.7424: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 21/100 | acc 0.7149 | loss 0.8217: 100%|██████████| 8/8 [00:00<00:00,  9.11it/s]

0.006

TRAIN | epoch 22/100 | acc 0.7972 | loss 0.5630: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 22/100 | acc 0.7424 | loss 0.7073: 100%|██████████| 8/8 [00:00<00:00,  9.20it/s]

0.006

0.7424 acc ile model kaydedildi

TRAIN | epoch 23/100 | acc 0.7981 | loss 0.5818: 100%|██████████| 32/32 [00:09<00:00,  3.49it/s]

TEST | epoch 23/100 | acc 0.7429 | loss 0.6174: 100%|██████████| 8/8 [00:00<00:00,  9.38it/s]

0.006

The model was saved with an accuracy of 0.7429.

TRAIN | epoch 24/100 | acc 0.8089 | loss 0.6961: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 24/100 | acc 0.7119 | loss 0.8060: 100%|██████████| 8/8 [00:00<00:00,  9.21it/s]

0.006

TRAIN | epoch 25/100 | acc 0.8053 | loss 0.5766: 100%|██████████| 32/32 [00:09<00:00,  3.47it/s]

TEST | epoch 25/100 | acc 0.7454 | loss 0.6883: 100%|██████████| 8/8 [00:00<00:00,  9.37it/s]

0.006

0.7454 acc ile model kaydedildi

TRAIN | epoch 26/100 | acc 0.8219 | loss 0.4906: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 26/100 | acc 0.7359 | loss 0.7066: 100%|██████████| 8/8 [00:00<00:00,  9.58it/s]

0.006

TRAIN | epoch 27/100 | acc 0.8315 | loss 0.8326: 100%|██████████| 32/32 [00:09<00:00,  3.46it/s]

TEST | epoch 27/100 | acc 0.7279 | loss 0.7797: 100%|██████████| 8/8 [00:00<00:00,  9.45it/s]

0.006

TRAIN | epoch 28/100 | acc 0.8392 | loss 0.6702: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 28/100 | acc 0.7284 | loss 0.7022: 100%|██████████| 8/8 [00:00<00:00,  9.30it/s]

0.006

TRAIN | epoch 29/100 | acc 0.8366 | loss 0.4542: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 29/100 | acc 0.7254 | loss 0.7145: 100%|██████████| 8/8 [00:00<00:00,  9.42it/s]

0.006

TRAIN | epoch 30/100 | acc 0.8502 | loss 0.3218: 100%|██████████| 32/32 [00:09<00:00,  3.46it/s]

TEST | epoch 30/100 | acc 0.7189 | loss 0.8417: 100%|██████████| 8/8 [00:00<00:00,  9.07it/s]

0.006

TRAIN | epoch 31/100 | acc 0.8630 | loss 0.4189: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 31/100 | acc 0.7174 | loss 0.6992: 100%|██████████| 8/8 [00:00<00:00,  9.43it/s]

0.006

TRAIN | epoch 32/100 | acc 0.8674 | loss 0.7398: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 32/100 | acc 0.6935 | loss 0.8833: 100%|██████████| 8/8 [00:00<00:00,  8.97it/s]

0.006

TRAIN | epoch 33/100 | acc 0.8687 | loss 0.4871: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 33/100 | acc 0.6955 | loss 0.8824: 100%|██████████| 8/8 [00:00<00:00,  8.89it/s]

0.006

TRAIN | epoch 34/100 | acc 0.8761 | loss 0.4460: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 34/100 | acc 0.7179 | loss 0.8441: 100%|██████████| 8/8 [00:00<00:00,  9.60it/s]

0.006

TRAIN | epoch 35/100 | acc 0.8843 | loss 0.4507: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 35/100 | acc 0.7349 | loss 0.7256: 100%|██████████| 8/8 [00:00<00:00,  9.70it/s]

0.006

TRAIN | epoch 36/100 | acc 0.8910 | loss 0.1809: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 36/100 | acc 0.7404 | loss 0.9161: 100%|██████████| 8/8 [00:00<00:00,  8.72it/s]

0.006

TRAIN | epoch 37/100 | acc 0.9043 | loss 0.4455: 100%|██████████| 32/32 [00:09<00:00,  3.49it/s]

TEST | epoch 37/100 | acc 0.7239 | loss 0.8318: 100%|██████████| 8/8 [00:00<00:00,  8.61it/s]

0.006

TRAIN | epoch 38/100 | acc 0.8860 | loss 0.4863: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 38/100 | acc 0.7244 | loss 0.7524: 100%|██████████| 8/8 [00:00<00:00,  9.49it/s]

0.006

TRAIN | epoch 39/100 | acc 0.8920 | loss 0.4202: 100%|██████████| 32/32 [00:09<00:00,  3.47it/s]

TEST | epoch 39/100 | acc 0.7339 | loss 0.9142: 100%|██████████| 8/8 [00:00<00:00,  8.72it/s]

0.006

TRAIN | epoch 40/100 | acc 0.9006 | loss 0.4741: 100%|██████████| 32/32 [00:09<00:00,  3.47it/s]

TEST | epoch 40/100 | acc 0.7299 | loss 0.7898: 100%|██████████| 8/8 [00:00<00:00,  9.61it/s]

0.006

TRAIN | epoch 41/100 | acc 0.9028 | loss 0.4016: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 41/100 | acc 0.7004 | loss 0.8195: 100%|██████████| 8/8 [00:00<00:00, 10.03it/s]

0.006

TRAIN | epoch 42/100 | acc 0.9125 | loss 0.2418: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 42/100 | acc 0.7234 | loss 0.7706: 100%|██████████| 8/8 [00:00<00:00,  8.35it/s]

0.006

TRAIN | epoch 43/100 | acc 0.9111 | loss 0.2926: 100%|██████████| 32/32 [00:09<00:00,  3.44it/s]

TEST | epoch 43/100 | acc 0.7264 | loss 0.7834: 100%|██████████| 8/8 [00:00<00:00,  9.45it/s]

0.006

TRAIN | epoch 44/100 | acc 0.9261 | loss 0.3431: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 44/100 | acc 0.7184 | loss 0.9836: 100%|██████████| 8/8 [00:00<00:00,  9.40it/s]

0.006

TRAIN | epoch 45/100 | acc 0.9165 | loss 0.2748: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 45/100 | acc 0.6885 | loss 1.0635: 100%|██████████| 8/8 [00:00<00:00,  9.76it/s]

0.006

TRAIN | epoch 46/100 | acc 0.9267 | loss 0.1266: 100%|██████████| 32/32 [00:09<00:00,  3.47it/s]

TEST | epoch 46/100 | acc 0.7124 | loss 0.8987: 100%|██████████| 8/8 [00:00<00:00,  8.66it/s]

0.006

TRAIN | epoch 47/100 | acc 0.9450 | loss 0.1705: 100%|██████████| 32/32 [00:09<00:00,  3.47it/s]

TEST | epoch 47/100 | acc 0.6370 | loss 1.1723: 100%|██████████| 8/8 [00:00<00:00,  9.48it/s]

0.006

TRAIN | epoch 48/100 | acc 0.9421 | loss 0.2057: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 48/100 | acc 0.7139 | loss 1.1116: 100%|██████████| 8/8 [00:00<00:00,  9.65it/s]

0.006

TRAIN | epoch 49/100 | acc 0.9252 | loss 0.2448: 100%|██████████| 32/32 [00:09<00:00,  3.46it/s]

TEST | epoch 49/100 | acc 0.6460 | loss 1.1030: 100%|██████████| 8/8 [00:00<00:00,  9.49it/s]

0.006

TRAIN | epoch 50/100 | acc 0.8742 | loss 0.2801: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 50/100 | acc 0.6855 | loss 1.0383: 100%|██████████| 8/8 [00:00<00:00,  9.26it/s]

0.006

TRAIN | epoch 51/100 | acc 0.9149 | loss 0.2426: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 51/100 | acc 0.6865 | loss 1.0759: 100%|██████████| 8/8 [00:00<00:00,  9.61it/s]

0.006

TRAIN | epoch 52/100 | acc 0.9307 | loss 0.1683: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 52/100 | acc 0.7324 | loss 0.9127: 100%|██████████| 8/8 [00:00<00:00,  9.72it/s]

0.006

TRAIN | epoch 53/100 | acc 0.9353 | loss 0.3262: 100%|██████████| 32/32 [00:09<00:00,  3.46it/s]

TEST | epoch 53/100 | acc 0.6990 | loss 1.1041: 100%|██████████| 8/8 [00:00<00:00,  9.17it/s]

0.006

TRAIN | epoch 54/100 | acc 0.9473 | loss 0.1243: 100%|██████████| 32/32 [00:09<00:00,  3.47it/s]

TEST | epoch 54/100 | acc 0.7264 | loss 0.9156: 100%|██████████| 8/8 [00:00<00:00,  9.81it/s]

0.006

TRAIN | epoch 55/100 | acc 0.9452 | loss 0.2656: 100%|██████████| 32/32 [00:09<00:00,  3.47it/s]

TEST | epoch 55/100 | acc 0.7224 | loss 0.9215: 100%|██████████| 8/8 [00:00<00:00,  9.81it/s]

0.006

TRAIN | epoch 56/100 | acc 0.9468 | loss 0.1721: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 56/100 | acc 0.7234 | loss 0.9548: 100%|██████████| 8/8 [00:00<00:00,  8.60it/s]

0.006

TRAIN | epoch 57/100 | acc 0.9366 | loss 0.2123: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 57/100 | acc 0.6266 | loss 5.0599: 100%|██████████| 8/8 [00:00<00:00, 10.13it/s]

0.006

TRAIN | epoch 58/100 | acc 0.9371 | loss 0.2581: 100%|██████████| 32/32 [00:09<00:00,  3.47it/s]

TEST | epoch 58/100 | acc 0.7159 | loss 0.9800: 100%|██████████| 8/8 [00:00<00:00,  9.33it/s]

0.006

TRAIN | epoch 59/100 | acc 0.9472 | loss 0.1915: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 59/100 | acc 0.6940 | loss 1.1127: 100%|██████████| 8/8 [00:00<00:00,  9.60it/s]

0.006

TRAIN | epoch 60/100 | acc 0.9497 | loss 0.3091: 100%|██████████| 32/32 [00:09<00:00,  3.49it/s]

TEST | epoch 60/100 | acc 0.7299 | loss 1.0300: 100%|██████████| 8/8 [00:00<00:00,  9.45it/s]

0.006

TRAIN | epoch 61/100 | acc 0.9493 | loss 0.2470: 100%|██████████| 32/32 [00:09<00:00,  3.48it/s]

TEST | epoch 61/100 | acc 0.7284 | loss 0.8383: 100%|██████████| 8/8 [00:00<00:00,  9.41it/s]

0.006

TRAIN | epoch 62/100 | acc 0.9548 | loss 0.2685: 100%|██████████| 32/32 [00:09<00:00,  3.49it/s]

TEST | epoch 62/100 | acc 0.7109 | loss 1.0505:  12%|█▎        | 1/8 [00:00<00:02,  3.38it/s]

  
**6. Results and Evaluation**

```

fig, ax = plt.subplots(1,3, figsize=(18,4), dpi=(96))

# Grafik (ACC)
ax[0].plot(torch.stack(train_acc).cpu())
ax[0].plot(torch.stack(test_acc).cpu())
ax[0].plot([highest_acc]*(len(train_acc)))
ax[0].scatter(torch.argmax(torch.tensor(test_acc)).item(), torch.tensor(test_acc).max().item(), s=128, c='red', marker='*')
           
ax[0].title.set_text('Accuracy')
ax[0].set(xlabel='epochs', ylabel='accuracy')
ax[0].legend(['train', 'val', f'highest {highest_acc:.4f}', f'best {best_acc:.4f}'], loc='upper left')

# Grafik (LOSS)
ax[1].plot(train_losses)
ax[1].plot(test_losses)
ax[1].plot([torch.tensor(test_losses).min()]*len(test_losses))
ax[1].scatter(torch.argmin(torch.tensor(test_losses)).item(), torch.tensor(test_losses).min().item(), s=128, c='red', marker='*')


ax[1].title.set_text('Loss')
ax[1].set(xlabel='epochs', ylabel='loss')
ax[1].legend(['train', 'val', f'lowest {torch.tensor(test_losses).min().item():.4f}', f'best {torch.tensor(test_losses).min().item():.4f}'], loc='upper left')

# Grafik (Learning Rate)
ax[2].plot(lrs)
ax[2].title.set_text('Learning Rate')
ax[2].set(xlabel='steps', ylabel='learning rate')

plt.show(block='off')

```


**Evaluating the Saved Best Model**


```
# Save the best accuracy checkpoint
best_model_check_point = torch.load('checkpoint.pt', map_location=DEVICE)
best_model = best_model_check_point['model']

preds = []
pred_prob = []
actual = []

criterion = nn.CrossEntropyLoss()

loader = tqdm(test_dl)

best_model.eval()
with torch.no_grad():
    count = 0
    correct = 0
    losses = 0
    
    for bidx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)

        actual.append(torch.flatten(labels))
        
        labels = labels.to(DEVICE)
        
        score = best_model(images)
        
        prob = F.softmax(score, dim=-1)
       
        pred = torch.argmax(score, dim=-1)
        
        preds.append(torch.flatten(pred))
        
        correct += (pred==labels).sum()
        
        count += len(labels)
    pred_prob.append(torch.flatten(prob))
        
        loss = eval_criterion(score, labels)
        
        losses += loss.item()
        
    print(f'ACC: {correct/count:.4f} | LOSS: {losses/len(test_dl)}')

```

**Confusion matrix**


```
for idx, p in enumerate(preds):
    if idx>0:
        preds_ = torch.cat([preds_, p], dim=0)
    else:
        preds_ = p
        
for idx, p in enumerate(actual):
    if idx>0:
        actual_ = torch.cat([actual_, p], dim=0)
    else:
        actual_ = p
        
conf_mat = confusion_matrix(actual_.cpu(), preds_.cpu())

```




```


sample_images = []
N = len(CLASSES)
for class_ in classes.keys():
    image_idxs = data.label==class_
    N_ = len(data[image_idxs])
    chosen = random.sample(list(np.arange(N_)), k= N)
    images = np.asarray(data[image_idxs].iloc[chosen,:-1])
    
    for img in images:
        sample_images.append(torch.tensor(img.reshape(28,28,3)).permute(2,0,1))
        
s = torch.stack(sample_images)
grid = torchvision.utils.make_grid(s, nrow=7, ncol=N)

plt.figure(figsize=(8,8), dpi=(128))
plt.imshow(grid.permute(1,2,0))
plt.xticks(np.linspace(14,grid.shape[2]-14,7), labels=[f'sample {idx}' for idx in range(N)])
plt.yticks(np.linspace(14,grid.shape[1]-14,7), labels=[f'[{idx}] {cls}' for idx, cls in enumerate(CLASSES_FULL)])
plt.title('Samples of skin lesions in HAM10000')
plt.legend(CLASSES_FULL)
plt.show(block='off')


```


```

cmn = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(12,10), dpi=(96))

plt.imshow(cmn, interpolation='nearest', cmap=plt.cm.Blues)

sns.heatmap(cmn, cmap='Blues', annot=True, fmt='.2f', xticklabels=[f'[{idx}]' for idx in range(len(CLASSES))], yticklabels=[f'[{idx}] {cls}' for idx, cls in enumerate(CLASSES_FULL)])
plt.ylabel('Gerçek')
plt.xlabel('Tahmin')
plt.title('Karmaşıklık Matrisi')
plt.show(block=False)

```

