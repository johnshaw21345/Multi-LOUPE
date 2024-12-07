import argparse
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

class Parameters():
    def __init__(self):
        super(Parameters, self).__init__()
        ## Hardware/GPU parameters =================================================
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.multi_GPU = True
        self.seed_num=42
        ## Network/Model parameters =================================================
        self.model_name = "UNetDC_short_first_rassp_ECA1"
        self.batch_size = 128
        self.activation_func = 'CReLU' # 'CReLU' # 'modReLU'  'ZReLU'
        self.w_grad_clip = 5
        self.lr = 0.01 #0.007 for unet || 0.0003 for unetdc
        self.epochs = 1
        self.loss_lambda = 0.3
        ## dataset prams ==============================================
        self.n_channels = 3
        self.mask_type = "loupe" #rad, rect, cartesian, loupe, full
        self.acc ="x4"
        self.rad_path = "./mask/rad"+ self.acc  + ".mat"
        self.threerad_path = "./mask/rad" + self.acc + "_3chan.mat"
        self.fiverad_path = "./mask/rad" + self.acc + "_5chan.mat"
        self.threerect_path = "./mask/car" + self.acc + "_3chansame.mat"        
        self.rect_path = "./mask/car"+ self.acc +".npy"
        self.loupe_pattern_path = "./mask/loupex8_auto_mask_axi.npy"
        self.train_path = "../data_recon"
        self.test_path = "../data_recon"
        self.workspace = '.'
        self.infer_path = "../output"
        self.save_weights = "../models"
        self.acc_name = 'kSpc0p4'
        self.Evaluation = False
        self.pre_trained = False
        self.pre_trained_model_path = "../cest_recon_result/x4rad_sagUNetDC_short_first_assp_allECA3_bs_32_ep_10_lr_0.03_msssim_lambda_1.0_kloss_lambda_0.0/UNetDC_short_first_assp_allECA3_LR_0.03_BS_32_best.pth"
        self.run_name = "x4rad_sagUNetDC_short_first_assp_allECA3_bs_32_ep_10_lr_0.03_msssim_lambda_1.0_kloss_lambda_0.0"
        # self.pre_trained_model_path = "../models/" +self.model_name+"/ResUnetPlusPlus_DC3_epoch39_LR_0.0003_BS_8_best.pth"
        self.msssim_lambda = 1.0
        self.kloss_lambda = 0.0
        ######### Dataset ####################
        self.training_percent = 0.8
        self.img_size = [96, 96]
        self.DC = True
        self.input_slices = list()
        self.num_slices_per_patient = list()
        self.groundTruth_slices = list()
        self.training_patients_index = list()
        self.us_rates = list()
        self.saveVolumeData = False

    def parse_args(self):
        parser = argparse.ArgumentParser(description='PyTorch Training')
        parser.add_argument('--data', metavar='DIR',  help='path to dataset')
        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('--epochs', default=1, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--b', '--batch-size', default=32, type=int,
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)')
        parser.add_argument('--acc', '-acceleration', default=4, type=int,
                            metavar='acc', help='mask acceleration (default: 4)')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        parser.add_argument('--lambd', default=0.0, type=float,
                            help='loss lambda 1')
        parser.add_argument('--kloss_lambda', default=1.0, type=float,
                            help='kloss_lambda lambda 1')
        parser.add_argument('--msssim_lambda', default=1.0, type=float,
                            help='msssim_lambda lambda 1')
        parser.add_argument('--no-save', '-n', action='store_false',
                            help='Do not save the output masks',
                            default=False)
        parser.add_argument('--model', '-m', default='MODEL_EPOCH417.pth',
                            metavar='FILE',
                            help='Specify the file in which is stored the model'" (default : 'MODEL.pth')")
        self.args = parser.parse_args()
        self.loss_lambda = self.args.lambd
        self.epochs = self.args.epochs
        self.lr = self.args.lr
        # self.acc = self.args.acc
        self.batch_size = self.args.b