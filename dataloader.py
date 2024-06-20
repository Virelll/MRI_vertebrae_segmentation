from mydataset import *
from torch.utils.data import DataLoader
from parameters import *


def createDataLoader(images, masks, transform, shuffle, num_classes):
    dataset = CustomTensorDataset(torch.from_numpy(images).type(torch.float32),
                                  torch.from_numpy(masks).type(torch.float32), num_classes, transform=transform)
    print("     create dataset successful\n")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print("     create dataloader successful\n")
    return dataloader


def createCatDataLoader(images_list, masks_list, transform, shuffle, num_classes):
    datasets = []
    for i in range(len(images_list)):
        datasets.append(CustomTensorDataset(torch.from_numpy(images_list[i]).type(torch.float32),
                                            torch.from_numpy(masks_list[i]).type(torch.float32), num_classes,
                                            transform=transform))
    res_dataset = torch.utils.data.ConcatDataset(datasets)
    dataloader = DataLoader(res_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# # train_dataset_320 = CustomTensorDataset(torch.from_numpy(img_train_320).type(torch.float32),torch.from_numpy(masks_train_320).type(torch.float32),count_classes_mask(masks_384),transform=train_transform)
# train_dataset = CustomTensorDataset(torch.from_numpy(img_train_384).type(torch.float32),torch.from_numpy(masks_train_384).type(torch.float32),count_classes_mask(masks_384),transform=train_transform)
# # train_dataset = torch.utils.data.ConcatDataset([train_dataset_320, train_dataset_384])
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
# # test_dataset_320 = CustomTensorDataset(torch.from_numpy(img_test_320).type(torch.float32),torch.from_numpy(masks_test_320).type(torch.float32),count_classes_mask(masks_384),transform=test_transform)
# test_dataset = CustomTensorDataset(torch.from_numpy(img_test_384).type(torch.float32),torch.from_numpy(masks_test_384).type(torch.float32),count_classes_mask(masks_384),transform=test_transform)
# # test_dataset = torch.utils.data.ConcatDataset([test_dataset_320, test_dataset_384])
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
