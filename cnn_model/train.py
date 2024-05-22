from OCR_Model import OCR_Model
import config
import torch
from torch.utils.data import DataLoader
from MyDataset import MyDataset
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter



if __name__ == "__main__":
    # Variables
    batch_size = 512
    num_workers = 2
    lr = 0.001
    writer = SummaryWriter("log")
    pre_epoch = 0
    device = ""
    module = input("Chọn chế độ：")
    epochs = int(input("Nhập số epochs："))
    if torch.cuda.is_available(): 
        device = "cuda"
    else:
        device = "cpu"
    
    # data loader
    train_dataset = MyDataset()
    train_dataset.getdata("../wb_recognition_dataset", "train")
    train_loader = DataLoader(train_dataset, batch_size,num_workers,pin_memory=True, drop_last=True)


    # config model
    model = OCR_Model().to(device)
    loss = nn.CrossEntropyLoss()
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr, weight_decay=1e-4)
    try:
        position = torch.load("./model.pt", map_location=device)
        model.load_state_dict(position["model"])
        pre_epoch = position["epoch"]
        optimizer.load_state_dict(position["optimizer"])
    except FileNotFoundError:
        print("Not download model!")
    if module == "train":
        model.train()  
    elif module == "val":
        model.eval()   

    # train
    for epoch in range(epochs):
        count = 0
        correct = 0
        countNum = 0
        for x,y in train_loader:
            countNum += 1
            print(countNum)
            count += len(y)
            pred = model(x.to(device))
            optimizer.zero_grad()
            loss_val = loss(pred, y.to(device)) 
            print("Loss is:", loss_val)
            if module == "train":
                loss_val.backward()
                optimizer.step()
            label = pred.argmax(1)
            for i in range(len(y)):
                if y[i] == label[i]:
                    correct += 1
        writer.add_scalar("Accuracy/Train", correct / count, epoch)
        print("Current epoch is :", epoch + pre_epoch + 1, " Accuracy is :", correct / count)
        state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1 + pre_epoch}
        if module == "train":
            torch.save(state_dict, "./model.pt")
    print("Finished!!!")
    writer.close()




