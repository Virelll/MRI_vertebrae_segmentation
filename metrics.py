from parameters import *


def Dice(labels, outputs, batch_size):
    output = torch.Tensor(np.argmax(outputs.detach().numpy(), axis=1)).byte()
    label = torch.Tensor(np.argmax(labels.detach().numpy(), axis=1)).byte()
    res_dice = 0
    for i in range(batch_size):
        dice = 0
        for l in np.unique(label[i].cpu().detach().numpy()):
            inter = output[i][label[i] == output[i]]
            dice += (inter[inter == l].shape[0] * 2 + 1) / (
                    output[i][output[i] == l].shape[0] + label[i][label[i] == l].shape[0] + 1)
        dice /= np.unique(label[i].cpu().detach().numpy()).shape[0]
        res_dice += dice
    return res_dice / batch_size


class SegLoss(nn.Module):
    def __init__(self):
        super(SegLoss, self).__init__()

    def forward(self, predict, target):
        return torch.mean(nn.CrossEntropyLoss(reduction='mean')(predict, target))


class MyDiceLoss(nn.Module):

    def __init__(self):
        super(MyDiceLoss, self).__init__()

    def forward(self, predict, target):
        output = torch.Tensor(np.argmax(predict.detach().numpy(), axis=1)).byte()
        label = torch.Tensor(np.argmax(target.detach().numpy(), axis=1)).byte()
        res_dice = 0
        for i in range(label.shape[0]):
            dice = 0
            for l in np.unique(label[i].cpu().detach().numpy()):
                inter = output[i][label[i] == output[i]]
                dice += (inter[inter == l].shape[0] * 2 + 1) / (
                        output[i][output[i] == l].shape[0] + label[i][label[i] == l].shape[0] + 1)
            dice /= np.unique(label[i].cpu().detach().numpy()).shape[0]
            res_dice += dice

        return 1 - res_dice / label.shape[0]
