import torch
from torch import nn

import numpy as np
from sklearn.preprocessing import LabelEncoder
import albumentations as A

train_transform = A.Compose([
    A.RandomCrop(width=304, height=304,p=0.5),
    A.Rotate(limit=(-20, 20),p=0.5),
    A.HorizontalFlip(p=0.2),
    A.CenterCrop(width=304, height=304,p=0.5),
    A.Resize(304,304),
])

test_transform = A.Compose([
    A.Resize(304,304),
])

def count_classes_mask(masks):
    labelencoder = LabelEncoder()
    n, h, w = masks.shape
    masks_reshaped = masks.reshape(-1,1)
    masks_reshaped_encoded = labelencoder.fit_transform(masks_reshaped)
    masks_encoded_original_shape = masks_reshaped_encoded.reshape(n, h, w)
    return len(np.unique(masks_encoded_original_shape))

batch_size = 2
torch.manual_seed(42)

print("     adding device....\n")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("     device add successful = ",device,"\n")
