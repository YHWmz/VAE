import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import argparse
import numpy as np
from VAE import VAE,VAE_CNN
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# 设定随机种子
def seed_torch(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Loss计算，注释掉的两行分别为L1 loss以及L2 Loss
def ELBO_loss(mean, logvar, x, x_new):
    KL = torch.sum(torch.exp(2 * logvar) / 2 + mean * mean / 2 - logvar - 0.5) / mean.shape[0]
    E = torch.nn.functional.binary_cross_entropy(x_new.view(-1, 28 * 28), x.view(-1, 28 * 28), reduction='sum') / \
        mean.shape[0]
    # E = torch.nn.functional.l1_loss(x, x_new, reduction='sum')/mean.shape[0]
    # E = torch.sum((x-x_new)*(x-x_new))/mean.shape[0]
    return KL + E, KL, E


def main(args):
    # 下载数据集
    seed_torch(args.seed)
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)

    # 载入数据集
    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # # iterator over train set
    # for batch_idx, (data, _) in enumerate(train_loader):
    #     print ("In train stage: data size: {}".format(data.size()))
    #     if batch_idx == 0:
    #         nelem = data.size(0)
    #         nrow  = 10
    #         save_image(data.view(nelem, 1, 28, 28), './images/image_0' + '.png', nrow=nrow)

    # # iterator over test set
    # for data, _ in test_loader:
    #     print ("In test stage: data size: {}".format(data.size()))

    # to be finished by you ...
    draw_res = args.draw_res
    recons_draw = args.recons_draw
    epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_dim =  [28 * 28, 1000, 512, 256] # [28 * 28, 1000, 256]
    latent_dim = args.latent_dim
    decoder_dim = [latent_dim, 256, 512, 1000, 784] # [latent_dim, 256, 1000, 784]
    model = VAE(encoder_dim, decoder_dim, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []

    # 开始训练
    for e in range(epochs):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            # 返回均值、标准差的对数、重建的image以及采样得到的隐向量z
            mean, logvar, x_new, z = model(x, device)
            loss, KL, E = ELBO_loss(mean, logvar, x, x_new)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

            # 对模型在test集的性能进行测试
            if batch_idx % 20 == 0:
                loss_test = []
                KL_test = []
                E_test = []
                for (x, _) in test_loader:
                    x = x.to(device)
                    mean, logvar, x_new, z = model(x, device)

                    loss, KL, E = ELBO_loss(mean, logvar, x, x_new)
                    loss_test.append(loss.item())
                    KL_test.append(KL.item())
                    E_test.append(E.item())

                print("epoch:{} \t reconstruction Loss:{} \t KLD Loss:{}, total Loss:{}".format(e, np.mean(E_test),
                                                                                                np.mean(KL_test),
                                                                                                np.mean(loss_test)))
        
        # 从[-5,5]中均匀采点，然后放入训练好的模型中进行图像的生成，并保存在相应文件夹下
        if draw_res:
            if args.latent_dim == 1:
                z_rand = np.arange(-5, 5, 0.5)
                z_rand = torch.tensor(z_rand, dtype=torch.float32).to(device).reshape(-1, 1)
                # z_rand = torch.randn(20, 1).to(device)
                nrow = int(z_rand.shape[0])
            elif args.latent_dim == 2:
                z_rand = np.zeros([20, 20, 2])
                z_tmp = np.arange(-2, 2, 0.2)
                for i in range(20):
                    z_rand[i, :, 0] = z_tmp
                    z_rand[:, i, 1] = z_tmp
                z_rand = torch.tensor(z_rand, dtype=torch.float32).to(device).reshape(400, 2)
                nrow = 20
            x_gen = model.generate(z_rand)
            # torch.save(model.state_dict(), './model2.h5')

            # 选择不同的文件夹保存图片  ./images/BCE_KL/
            pic_name = args.store_path + 'image_' + str(e)
            save_image(x_gen, pic_name + '.png', nrow=nrow)

        # 绘制0~9数字的重建结果，得到compare_raw与compare_gen两个文件，前者为原图，后者为VAE输出的图片
        if recons_draw:
            l = 0
            x_raw = []
            x_gen = []
            for (x, label) in test_loader:
                x = x.to(device)
                mean, logvar, x_new, z = model(x, device)

                for idx in range(len(label)):
                    if label[idx] == l:
                        x_raw.append(x[idx])
                        x_gen.append(x_new[idx])
                        l += 1
                        if l == 10:
                            break

            x_raw = torch.tensor([item.cpu().detach().numpy() for item in x_raw])
            x_gen = torch.tensor([item.cpu().detach().numpy() for item in x_gen])
            pic_name = args.store_path + 'compare_'
            nrow = 10
            save_image(x_raw, pic_name + 'raw' + '.png', nrow=nrow)
            save_image(x_gen, pic_name + 'gen' + '.png', nrow=nrow)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 进行参数的设置
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--draw_res", type=bool, default=True)
    parser.add_argument("--recons_draw", type=bool, default=True)
    parser.add_argument("--store_path", type=str, default="./images/final_try/")
    args = parser.parse_args()

    main(args)
