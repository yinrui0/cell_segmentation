import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from unet import UNet
from bowl_dataset import *
import torch.nn.functional as F
import numpy as np
import cv2


def dice_loss(inputs, target):
    smooth = 1.
    input_flatten, target_flatten = inputs.view(-1), target.view(-1)
    dice_loss = .0 - (((2. * (input_flatten * target_flatten).sum() + smooth) /
        (input_flatten.sum() + target_flatten.sum() + smooth)))
    #print(dice_loss)

    return dice_loss

def calculate_val_accuracy(val_data_loader, model, threshold=0.5, is_show=False):

    accuracy = 0.0
    for i, batch_data in enumerate(val_data_loader):

        batch_input = Variable(batch_data['image'].cuda())
        batch_gt_mask = Variable(batch_data['mask'].cuda())
        pred_mask = model(batch_input)
        accuracy += - dice_loss(F.sigmoid(pred_mask), batch_gt_mask).data[0]
        if is_show:
            show_plot(batch_input[0], batch_gt_mask[0], F.sigmoid(pred_mask[0]), threshold, "val_"+str(i))
    total = len(val_data_loader)

    return accuracy/total


def show_plot(image, gt_mask, pred_mask, threshold=0.5, label="none"):
    if gt_mask is not None:
        image_d = image.cpu().data.numpy()
        image_d = np.transpose(image_d, [1,2,0]) * 255.0
        gt_mask_np = gt_mask.cpu().data.numpy()
        gt_mask_np = np.transpose(gt_mask_np, [1,2,0]) * 255.0
        gt_mask_np = np.repeat(gt_mask_np, 3, 2)
        pred_mask = pred_mask.cpu().data.numpy()
        pred_mask[pred_mask>threshold] = 1
        pred_mask[pred_mask<threshold] = 0
        pred_mask = np.transpose(pred_mask, [1,2,0]) * 255.0
        pred_mask = np.repeat(pred_mask, 3, 2)
        concat_img = np.concatenate((image_d, gt_mask_np, pred_mask), axis=1)

        cv2.imwrite('./temp_result/pred_img_'+label+'.jpg', concat_img)
    else:
        image_d = image.cpu().data.numpy()
        image_d = np.transpose(image_d, [1,2,0]) * 255.0
        pred_mask = pred_mask.cpu().data.numpy()
        #print(pred_mask)
        #print(np.max(pred_mask))
        pred_mask[pred_mask>threshold] = 1
        pred_mask[pred_mask<threshold] = 0
        pred_mask = np.transpose(pred_mask, [1,2,0]) * 255.0
        pred_mask = np.repeat(pred_mask, 3, 2)
        concat_img = np.concatenate((image_d, pred_mask), axis=1)
        #print(concat_img.shape)

        cv2.imwrite('./temp_result/pred_img_'+label+'.jpg', concat_img)

    # cv2.imshow('i', img.astype(np.uint8))
    # cv2.waitKey(10)



def train(train_image, train_mask, test_image):
    epochs = 30
    batch_size = 20
    lr = 0.001

    print("loading training data......")
    train_transform = Compose([Rescale(132), RandomCrop(128), RandomHorizontallyFlip(), ToTensor()])
   
    train_dataset = BowlDataset(train_image, train_mask, test_image, fold="train", transform=train_transform)
    train_data_loader = DataLoader(train_dataset, batch_size, True, num_workers=4)
    print("Train set size: "+str(len(train_dataset)))

    val_dataset = BowlDataset(train_image, train_mask, test_image, fold="val", transform=train_transform)
    val_data_loader = DataLoader(val_dataset, batch_size, True, num_workers=4)
    print("Val set size: "+str(len(val_dataset)))


    model = UNet().cuda()


    criterion = torch.nn.BCEWithLogitsLoss(size_average=True)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-2)

    train_loss_over_epochs = []
    val_accuracy_over_epochs = []

    epoch = 0
    forward_times = 0

    for epoch in range(epochs):

        running_loss = 0.0
        if (epoch + 1) % 10 ==0:
            lr = lr * 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for i, batch_data in enumerate(train_data_loader):

            batch_input = Variable(batch_data['image'].cuda())
            batch_gt_mask = Variable(batch_data['mask'].cuda())

            optimizer.zero_grad()

            pred_mask = model(batch_input)

            ### show plot
            if (i+1) % 1 == 0:
                show_plot(batch_input[0], batch_gt_mask[0], F.sigmoid(pred_mask[0]), threshold=0.5, label="train_"+str(i+1))

            loss = criterion(pred_mask, batch_gt_mask)
            loss += dice_loss(F.sigmoid(pred_mask), batch_gt_mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            
        running_loss/=len(train_data_loader)
        print('[%d] loss: %.3f' % (epoch + 1, running_loss))

        val_accuracy = calculate_val_accuracy(val_data_loader, model, threshold=0.5, is_show=True)
        print('Negative dice loss of the network on the val images: %.5f ' % (val_accuracy))

        train_loss_over_epochs.append(running_loss)
        val_accuracy_over_epochs.append(val_accuracy)

        if (epoch+1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
            torch.save(checkpoint, './checkpoint/unet-{}'.format(epoch+1))

    return model

def test(model, train_image, train_mask, test_image, threshold=0.5):
    batch_size = 1
    test_transform = Compose([ToTensor()])

    test_dataset = BowlDataset(train_image, train_mask, test_image, fold="test", transform=test_transform)
    test_data_loader = DataLoader(test_dataset, batch_size, True, num_workers=4)
    print("Test set size: "+str(len(test_dataset)))

    model.eval()

    # batch_inputs = []
    # pred_masks = []
    for i, batch_data in enumerate(test_data_loader):

        batch_input = Variable(batch_data['image']).cuda()
        pred_mask = model(batch_input)
        pred_mask = F.sigmoid(pred_mask)
        show_plot(batch_input[0], None, pred_mask[0], 0.6, "test_"+str(i))
        #batch_inputs.append(batch_input)
        #pred_masks.append(pred_mask)

    #return batch_inputs, pred_masks



if __name__ == '__main__':
    train_image = np.load("./data/train_image.npy")
    train_mask = np.load("./data/train_mask.npy")
    test_image = np.load("./data/test_image.npy")

    model = train(train_image, train_mask, test_image)
    test(model, train_image, train_mask, test_image)
    #batch_inputs, pred_masks = test(model, train_image, train_mask, test_image)
    #np.save("batch_inputs",np.array(batch_inputs))
    #np.save("pred_masks",np.array(pred_masks))
    # show_plot(batch_inputs[0][0], None, pred_masks[0][0], 0.5, "test")
    #show_plot(batch_inputs[0][1], None, pred_masks[0][1], 0.5, "test")




