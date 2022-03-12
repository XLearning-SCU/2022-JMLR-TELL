import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torchvision.datasets.mnist import MNIST
from model import NetConv
from evaluation import evaluate, evaluate_others
import numpy as np
import matplotlib.pyplot as plt
import os


def save_model(model, optimizer, current_epoch):
    out = os.path.join("./save/checkpoint_{}.tar".format(current_epoch))
    state = {
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": current_epoch,
    }
    torch.save(state, out)


def inference():
    net.compute_cluster_center(alpha)
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
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    pred_vector = np.array(pred_vector)
    return feature_vector, labels_vector, pred_vector


def visualize_cluster_center():
    with torch.no_grad():
        cluster_center = net.compute_cluster_center(alpha)
        reconstruction = net.decode(cluster_center)

    plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(
            reconstruction[i]
            .detach()
            .cpu()
            .numpy()
            .reshape(dataset[0][0].shape[1], dataset[0][0].shape[2]),
            cmap="gray",
        )
    plt.savefig("./cluster_center.png")
    plt.close()


if __name__ == "__main__":
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    reload = False
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    train_dataset = MNIST(
        root="./datasets", train=True, download=True, transform=transforms
    )
    test_dataset = MNIST(
        root="./datasets", train=False, download=True, transform=transforms
    )
    dataset = ConcatDataset([train_dataset, test_dataset])
    class_num = 10
    batch_size = 256
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    data_loader_test = DataLoader(
        dataset, batch_size=500, shuffle=False, drop_last=False
    )
    net = NetConv(channel=1, inner_dim=784, class_num=class_num).cuda()
    optimizer = torch.optim.Adadelta(net.parameters())
    criterion = nn.MSELoss(reduction="mean")
    start_epoch = 0
    epochs = 3001
    alpha = 0.001
    net.normalize_cluster_center(alpha)
    if reload:
        model_fp = os.path.join("./save/checkpoint_3000.tar")
        checkpoint = torch.load(model_fp)
        net.load_state_dict(checkpoint["net"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
    for epoch in range(start_epoch, epochs):
        loss_clu_epoch = loss_rec_epoch = 0
        net.train()
        for step, (x, y) in enumerate(data_loader):
            x = x.cuda()
            z = net.encode(x)

            if epoch % 2 == 1:
                cluster_batch = net.cluster(z)
            else:
                cluster_batch = net.cluster(z.detach())
            soft_label = F.softmax(cluster_batch.detach(), dim=1)
            hard_label = torch.argmax(soft_label, dim=1)
            delta = torch.zeros((batch_size, 10), requires_grad=False).cuda()
            for i in range(batch_size):
                delta[i, torch.argmax(soft_label[i, :])] = 1
            loss_clu_batch = 2 * alpha - torch.mul(delta, cluster_batch)
            loss_clu_batch = 0.01 / alpha * loss_clu_batch.mean()

            x_ = net.decode(z)
            loss_rec = criterion(x, x_)

            loss = loss_rec + loss_clu_batch
            optimizer.zero_grad()
            loss.backward()
            if epoch % 2 == 0:
                net.cluster_layer.weight.grad = (
                    F.normalize(net.cluster_layer.weight.grad, dim=1) * 0.2 * alpha
                )
            else:
                net.cluster_layer.zero_grad()
            optimizer.step()
            net.normalize_cluster_center(alpha)
            loss_clu_epoch += loss_clu_batch.item()
            loss_rec_epoch += loss_rec.item()
        print(
            f"Epoch [{epoch}/{epochs}]\t Clu Loss: {loss_clu_epoch / len(data_loader)}\t Rec Loss: {loss_rec_epoch / len(data_loader)}"
        )

        if epoch % 50 == 0:
            visualize_cluster_center()
            feature, label, pred = inference()
            nmi, ari, acc = evaluate(label, pred)
            print("Model NMI = {:.4f} ARI = {:.4f} ACC = {:.4f}".format(nmi, ari, acc))
            ami, homo, comp, v_mea = evaluate_others(label, pred)
            print(
                "Model AMI = {:.4f} Homogeneity = {:.4f} Completeness = {:.4f} V_Measure = {:.4f}".format(
                    ami, homo, comp, v_mea
                )
            )
            save_model(net, optimizer, epoch)
