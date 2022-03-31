import pdb
import argparse
import torch
import torch.backends.cudnn as cudnn

from config import *
from dataset import *
from models import *
from utils import *


def test(args):

    # Device Init
    device = config.device
    cudnn.benchmark = True

    # Data Load
    testloader = data_loader(args, mode='test')

    # Model Load
    net, _, _, _ = load_model(args, class_num=config.class_num, mode='test')

    net.eval()
    torch.set_grad_enabled(False)
    for idx, (inputs, paths) in enumerate(testloader):
        inputs = inputs.to(device)
        outputs = net(inputs)
        if type(outputs) == tuple:
            outputs = outputs[0]
        post_process(args, inputs, outputs, paths)
        print(idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default='pspnet_res18',
                        help="Model Name (unet, pspnet_squeeze, pspnet_res50,\
                        pspnet_res34, pspnet_res50, deeplab)")
    parser.add_argument("--batch_size", type=int, default=18,
                        help="The batch size to load the data")
    parser.add_argument("--data", type=str, default="complete",
                        help="Label data type.")
    parser.add_argument("--img_root", type=str, default="/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/2020/MICCAI_BraTS_2018_Data_Training/test/LGG/image_T1",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--output_root", type=str, default="./output/test_LGG_prediction_pspnet_res18",
                        help="The directory containing the results.")
    parser.add_argument("--ckpt_root", type=str, default="./checkpoint/LGG",
                        help="The directory containing the trained model checkpoint")
    args = parser.parse_args()

    test(args)
