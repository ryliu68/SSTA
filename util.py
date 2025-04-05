# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


def distance(x_adv, x, norm='l2'):
    diff = (x_adv - x).contiguous().view(x.size(0), -1)
    if norm == 'l2':
        out = torch.sqrt(torch.sum(diff * diff)).item()
        return out
    elif norm == 'linf':
        out = torch.sum(torch.max(torch.abs(diff), 1)[0]).item()
        return out


def flow_st(images, flows):

    batch_size, _, H, W = images.size()

    device = images.device

    # basic grid: tensor with shape (2, H, W) with value indicating the
    # pixel shift in the x-axis or y-axis dimension with respect to the
    # original images for the pixel (2, H, W) in the output images,
    # before applying the flow transforms
    grid_single = torch.stack(
        torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    ).float()

    grid = grid_single.repeat(batch_size, 1, 1, 1)  # 100,2,28,28

    images = images.permute(0, 2, 3, 1)  # 100, 28,28,1

    grid = grid.to(device)

    grid_new = grid + flows
    # assert 0

    sampling_grid_x = torch.clamp(
        grid_new[:, 1], 0., (W - 1.)
    )
    sampling_grid_y = torch.clamp(
        grid_new[:, 0], 0., (H - 1.)
    )

    # now we need to interpolate

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a square around the point of interest
    x0 = torch.floor(sampling_grid_x).long()
    x1 = x0 + 1
    y0 = torch.floor(sampling_grid_y).long()
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate image boundaries
    # - 2 for x0 and y0 helps avoiding black borders
    # (forces to interpolate between different points)
    x0 = torch.clamp(x0, 0, W - 2)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 2)
    y1 = torch.clamp(y1, 0, H - 1)

    b = torch.arange(0, batch_size).view(
        batch_size, 1, 1).repeat(1, H, W).to(device)
    # assert 0
    Ia = images[b, y0, x0].float()
    Ib = images[b, y1, x0].float()
    Ic = images[b, y0, x1].float()
    Id = images[b, y1, x1].float()

    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()

    wa = (x1 - sampling_grid_x) * (y1 - sampling_grid_y)
    wb = (x1 - sampling_grid_x) * (sampling_grid_y - y0)
    wc = (sampling_grid_x - x0) * (y1 - sampling_grid_y)
    wd = (sampling_grid_x - x0) * (sampling_grid_y - y0)

    # add dimension for addition
    wa = wa.unsqueeze(3)
    wb = wb.unsqueeze(3)
    wc = wc.unsqueeze(3)
    wd = wd.unsqueeze(3)

    # compute output
    perturbed_image = wa * Ia + wb * Ib + wc * Ic+wd * Id

    perturbed_image = perturbed_image.permute(0, 3, 1, 2)

    return perturbed_image


def get_model(args):
    if args.dataset == "imagenet":
        if args.net == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif args.net == "resnet34":
            net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif args.net == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.net == "resnet152":
            net = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        elif args.net == "vgg16":
            net = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        elif args.net == "vgg19":
            net = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        elif args.net == "mobilenetv2":
            net = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT)
        elif args.net == "densenet121":
            net = models.densenet121(
                weights=models.DenseNet121_Weights.DEFAULT)
        elif args.net == "densenet169":
            net = models.densenet169(
                weights=models.DenseNet169_Weights.DEFAULT)
        elif args.net == "inceptionv3":
            net = models.inception_v3(
                weights=models.Inception_V3_Weights.DEFAULT)
        elif args.net == "vit16":
            net = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        elif args.net == "vit32":
            net = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
        elif args.net == "swin_b":
            net = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)

        else:
            raise NotImplementedError

    else:
        raise ValueError

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize = NormalizeByChannelMeanStd(
        mean=mean, std=std)
    net = nn.Sequential(
        normalize,
        net
    )

    net.eval()

    return net
