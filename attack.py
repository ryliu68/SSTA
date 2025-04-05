import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Resize
from tqdm import tqdm

from TRACER.config import getConfig
from TRACER.model.TRACER import TRACER
from TRACER.util.utils import load_pretrained
from util import flow_st, get_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'


resizer = Resize([320, 320])


def cw_loss(logits, labels, targeted=False):
    onehot_labels = torch.eye(1000, device=labels.device)[
        labels]
    real = torch.sum(onehot_labels * logits, dim=1)
    other, _ = torch.max((1 - onehot_labels) *
                         logits - onehot_labels * 10000, dim=1)
    zeros = torch.zeros_like(other)
    if targeted:
        adv_loss = torch.max(other-real, zeros)
    else:
        adv_loss = torch.max(real-other, zeros)
    loss = torch.mean(adv_loss)

    return loss


def attack(args, data):

    tracer_args = getConfig()

    tracer_model = TRACER(tracer_args).to(device)

    path = load_pretrained(f'TE-{tracer_args.arch}')

    weights_dict = {}
    for k, v in path.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    tracer_model.load_state_dict(weights_dict)
    print('###### pre-trained Model restored #####')

    # data
    _images = data["images"]
    _labels = data["labels"]
    data_len = _labels.size(0)

    # model
    tracer_model.eval()

    classifier = get_model(args)

    classifier.to(device)
    classifier.eval()

    #
    success = 0
    test_imgs = 0
    adv_imgs = None

    for batch_idx in tqdm(range(data_len)):
        bound = 180

        image = _images[batch_idx].to(device)
        label = _labels[batch_idx].to(device)

        target_label = torch.zeros_like(label)

        image = image.unsqueeze(0)

        orig_image = image.clone()

        pre_label = torch.argmax(classifier(orig_image))

        if pre_label != label:
            continue

        re_image = resizer(image)

        output, edge_mask, ds_map = tracer_model(
            (re_image-args.means)/args.stds)

        output = F.interpolate(output, size=(224, 224), mode='bilinear')

        output = output.squeeze(0).squeeze(0)

        mask = output.ge(bound/255)

        mask_size = mask.sum().item()

        if mask_size <= 2500:
            continue

        test_imgs += 1

        image = image*mask

        flow_field = torch.zeros([1, 2, 224, 224]).to(device)
        flow_field.requires_grad = True

        optimizer = torch.optim.Adam([flow_field], lr=args.lr)

        succ_flag = False
        for n in tqdm(range(250), disable=True):
            if n % 20 == 0:
                bound -= 10
                mask = output.ge(bound/255)
                image = image*mask

            flowed_img = flow_st(image, torch.clamp(flow_field, -1., 1.))

            adv_img = torch.where(flowed_img != 0, flowed_img, orig_image)

            out = classifier(adv_img)

            if args.targeted:
                loss = cw_loss(out, target_label, targeted=True)
                if torch.argmax(out) == target_label:
                    succ_flag = True
                    break
            else:
                loss = cw_loss(out, label)
                if torch.argmax(out) != pre_label:
                    succ_flag = True
                    break

            optimizer.zero_grad()
            loss.backward(loss.clone().detach())
            optimizer.step()

        if succ_flag:
            success += 1
            if args.net == "resnet50" and batch_idx <= 10:
                if adv_imgs is None:
                    imgs = orig_image.detach().cpu()
                    labels = label.detach().cpu()
                    adv_imgs = adv_img.detach().cpu()
                    # adv_labels = adv_label.detach().cpu()
                else:
                    imgs = torch.vstack((imgs, orig_image.detach().cpu()))
                    labels = torch.hstack((labels, label.detach().cpu()))
                    adv_imgs = torch.vstack(
                        (adv_imgs, adv_img.detach().cpu()))
                    # adv_labels = torch.hstack(
                    #     (adv_labels, adv_label.detach().cpu()))

    print("Attack Success Rate: {:.2f} %".format(
        success*100/test_imgs))

    if adv_imgs is not None:
        state = {
            'adv_imgs': adv_imgs,
            'imgs': imgs,
            'labels': labels
        }

        print(adv_imgs.size())
        if args.targeted:
            save_dir = F"adv_output/target/SSTA"
        else:
            save_dir = F"adv_output/untarget/SSTA"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(state, F"{save_dir}/{args.dataset}_{args.net}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net',  default="resnet50",
                        type=str, help='')
    parser.add_argument('-num', '--data_num',  default=100,
                        type=int, help='data number for attack')
    parser.add_argument('-D', '--dataset',  default="imagenet",
                        type=str, help='')
    parser.add_argument('-LR', '--lr',  default=0.01,
                        type=float, help='learning_rate')
    parser.add_argument('-T', '--targeted',  default=False,
                        type=bool, help='learning_rate')

    args = parser.parse_args()

    if args.dataset == "imagenet":
        args.means = torch.Tensor([0.485, 0.456, 0.406]).view(
            [1, 3, 1, 1]).to(device)
        args.stds = torch.Tensor([0.229, 0.224, 0.225]).view(
            [1, 3, 1, 1]).to(device)

    data = torch.load("demo_data.pth")  # we only provide 10 image-label pairs for demo use

    _models = ["vgg19", "resnet50", "densenet121", "vit16", "swin_b"]

    for net in _models:
        args.net = net

        print(F"\nAttacking {net}: \n")

        attack(args, data)
