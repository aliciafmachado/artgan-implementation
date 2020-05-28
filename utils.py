import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt


def gen_z(batch_size, d):
    return torch.randn(batch_size, d)


def gen_yk(batch_size, num_classes):
    t = torch.zeros(batch_size, num_classes)
    for i, row in enumerate(t):
        num = random.randint(0, num_classes - 1)
        # num = i % num_classes
        row[num] = 1
    return t


def gen_noise(batch_size, epoch):
  return torch.normal(0.0, 0.1/(epoch+1), (batch_size, 3, 64, 64))


def fake_v(batch_size, num_classes):
    v = torch.zeros(batch_size) + num_classes
    return v


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=80):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def gen_y_test(n_classes, test_num=10):
    # defined only for multiple of 10
    z = torch.randn(test_num, 100)
    y = torch.zeros(test_num, n_classes)
    l = []
    for i, row in enumerate(y):
        row[i % n_classes] = 1
        l.append(i % n_classes)
    return torch.cat([z, y], 1).type(torch.cuda.FloatTensor), l


def save_img(G, D, epoch, classes, test_num=20, path=None):
    G.eval()
    D.eval()
    path_img = path + "/Wikiart_images/epoch_" + str(epoch)
    if not os.path.exists(path_img):
        os.makedirs(path_img)

    print("len - 1")
    print(len(classes) - 1)
    print("test_num")
    print(test_num)
    y, l = gen_y_test(len(classes)-1 , test_num=test_num)
    imgs = G(y)
    output = D(imgs)
    probs, predicted = torch.max(output.data, 1)

    z = zip(l, imgs, probs, predicted)
    for i, values in enumerate(z):
        label, img_solo, p, pred = values
        print()
        print("Image %d" % i)
        npimg = img_solo.cpu().detach().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        path_img_save = path_img + "/Image_" + str(i) + "_" + classes[label] + "_" + classes[pred.item()] + ".jpg"
        plt.savefig(path_img_save)
        plt.show()
        print("label = {} ({})".format(classes[label], label))
        print("Calculated label by D = {} ({}) -> Prob = {:02.1f}".format(classes[pred.item()], pred.item(), 100 * p))

    G.train()
    D.train()
