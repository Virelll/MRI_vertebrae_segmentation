from tqdm.auto import tqdm
import parameters
from metrics import *
from torchsummary import summary

class Net():
    def __init__(self,model,optimizer='Adam',lr=0.001,epochs=1):
        super(Net, self).__init__()
        print("     creating model...\n")
        self.model = model.to(device)
        print("     model created successful\n")

        summary(self.model, (1,304,304))

        self.lr = lr
        self.epochs = epochs

        if optimizer == 'Adam':
            print("     creating optimizer Adam...\n")
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
            print("     Adam created successful\n")
        elif optimizer == 'Adamax':
            print("     creating optimizer Adamax...\n")
            self.optimizer = torch.optim.Adamax(params=self.model.parameters(), lr=self.lr)
            print("     Adamax created successful\n")


        self.history = {'train_dice_list': [], 'train_loss_list': [],'train_loss1_list':[],'train_loss2_list':[],
                        'test_dice_list': [], 'test_loss_list': [],'test_loss1_list':[],'test_loss2_list':[]}

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max',
                                                                    patience=5, factor=0.5)

    def train(self,train_dataloader, test_dataloader):
        loss_fn_1 = MyDiceLoss()
        loss_fn_2 = SegLoss()

        for epoch in tqdm(range(self.epochs)):
            print(f"Epoch: {epoch + 1} of {self.epochs}")

            # Training
            train_loss_1, train_loss_2, train_loss, train_acc, train_dice = 0, 0, 0, 0, 0

            self.model.train()
            for batch, (X, y) in enumerate(train_dataloader):
                X, y = X.to(device), y.to(device)

                y_pred = self.model(X)

                loss_1 = loss_fn_1(y_pred.cpu(), y.cpu())  # DiceLoss
                loss_2 = loss_fn_2(y_pred, y)  # SegLoss
                loss = loss_1 + loss_2

                train_loss += loss
                train_loss_1 += loss_1
                train_loss_2 += loss_2
                train_dice += Dice(y.cpu(), y_pred.cpu(), y_pred.shape[0])

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

            train_loss /= len(train_dataloader)
            train_loss_1 /= len(train_dataloader)
            train_loss_2 /= len(train_dataloader)
            train_dice /= len(train_dataloader)

            self.history['train_dice_list'].append(train_dice)
            self.history['train_loss_list'].append(train_loss)
            self.history['train_loss1_list'].append(train_loss_1)
            self.history['train_loss2_list'].append(train_loss_2)

            # Testing
            test_loss_1, test_loss_2, test_loss, test_acc, test_dice = 0, 0, 0, 0, 0
            self.model.eval()
            with torch.inference_mode():
                for X, y in test_dataloader:
                    X, y = X.to(device), y.to(device)

                    y_pred = self.model(X)

                    loss_1 = loss_fn_1(y_pred.cpu(), y.cpu())  # DiceLoss
                    loss_2 = loss_fn_2(y_pred, y)  # SegLoss
                    loss = loss_1 + loss_2
                    test_loss += loss
                    test_loss_1 += loss_1
                    test_loss_2 += loss_2
                    test_dice += Dice(y.cpu(), y_pred.cpu(), y_pred.shape[0])

                test_loss /= len(test_dataloader)
                test_loss_1 /= len(test_dataloader)
                test_loss_2 /= len(test_dataloader)
                test_acc /= len(test_dataloader)
                test_dice /= len(test_dataloader)

                self.history['test_dice_list'].append(test_dice)
                self.history['test_loss_list'].append(test_loss)
                self.history['test_loss1_list'].append(test_loss_1)
                self.history['test_loss2_list'].append(test_loss_2)

                self.scheduler.step(test_dice)

            print(f"Train Dice: {train_dice:.5f} | Train loss: {train_loss:.5f}, "
                  f"DiceLoss: {train_loss_1:.5f}, SegLoss: {train_loss_2:.5f} | "
                  f"Test dice: {test_dice:.5f} | Test loss: {test_loss:.5f}, "
                  f"DiceLoss: {test_loss_1:.5f}, SegLoss: {test_loss_2:.5f}\n")

    def save_model(self,path):
        torch.save(self.model, path)

    def load_model(self,path):
        self.model = torch.load(path).to(device) # "D:\\MRI\\C_model.pt"
