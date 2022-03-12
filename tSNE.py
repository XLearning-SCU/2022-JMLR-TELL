from pickle import FALSE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torchvision.datasets.mnist import MNIST
from model import NetConv

np.random.seed(42)
torch.manual_seed(42)


def tSNE(X, Y, epoch):
    colors = list(mcolors.TABLEAU_COLORS.keys())
    print("Performing TSNE")
    X_embedded = TSNE(n_components=2).fit_transform(X)
    plt.figure()
    N = X_embedded.shape[0] - class_num
    for i in range(N):
        if i % 1000 == 0:
            print(i, "/", N)
        label = Y[i].item()
        plt.scatter(
            X_embedded[i][0],
            X_embedded[i][1],
            color=mcolors.TABLEAU_COLORS[colors[label]],
            marker=".",
            s=1,
        )
    for i in range(class_num):
        plt.scatter(
            X_embedded[i + N][0], X_embedded[i + N][1], color="black", marker="*", s=50
        )
    plt.savefig("tSNE_epoch=" + str(epoch) + ".png")
    print("Finished")


def inference():
    net.eval()
    feature_vector = []
    labels_vector = []
    pred_vector = []
    with torch.no_grad():
        for step, (x, y) in enumerate(data_loader_test):
            x = x.cuda()
            with torch.no_grad():
                z = net.encode(x)
                pred = net.predict(z)
            feature_vector.extend(z.detach().cpu().numpy())
            labels_vector.extend(y.numpy())
            pred_vector.extend(pred.detach().cpu().numpy())
    return feature_vector, labels_vector, pred_vector


if __name__ == "__main__":
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    train_dataset = MNIST(
        root="./datasets", train=True, download=True, transform=transforms
    )
    test_dataset = MNIST(
        root="./datasets", train=False, download=True, transform=transforms
    )
    dataset = test_dataset  # ConcatDataset([train_dataset, test_dataset])
    class_num = 10
    batch_size = 256
    alpha = 0.001
    data_loader_test = DataLoader(
        dataset, batch_size=500, shuffle=False, drop_last=False
    )
    net = NetConv(channel=1, inner_dim=784, class_num=class_num).cuda()
    epochs = [50, 100, 250, 500, 1000, 3000]
    for epoch in epochs:
        model_fp = os.path.join("./save/checkpoint_{}.tar".format(epoch))
        checkpoint = torch.load(model_fp)
        net.load_state_dict(checkpoint["net"], strict=False)

        print("Computing features from model")
        feature, labels, pred = inference()
        cluster_center = net.compute_cluster_center(alpha)
        feature.extend(cluster_center.detach().cpu().numpy())
        feature_vector = np.array(feature)
        labels_vector = np.array(labels)
        tSNE(feature_vector, labels_vector, epoch)
