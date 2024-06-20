from torch.utils.data import Dataset
import numpy as np

class CustomTensorDataset(Dataset):
    def __init__(self, x, y, num_classes, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
        self.num_classes = num_classes

    def reshape_mask(self, mask):
        masks_cat = np.eye(self.num_classes)[mask.astype("uint8")]
        masks_cat = np.transpose(masks_cat, (2, 0, 1)).astype("float32")
        return masks_cat

    def __getitem__(self, index):
        x_res = self.x[index][0]
        y_res = self.y[index][0]

        if self.transform:
            transformed = self.transform(image=np.array(x_res, dtype='float32'), mask=np.array(y_res, dtype='float32'))
            x_res = transformed['image']
            y_res = self.reshape_mask(transformed['mask'])

        return x_res.reshape((1, 304, 304)), y_res

    def __len__(self):
        return self.x.shape[0]