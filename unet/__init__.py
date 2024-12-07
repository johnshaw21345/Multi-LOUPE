from .unet_model import UNet,UNetDC,UNetDC_short, UNetDC_short_ECA,UNetDC_IK, UNetunetDC_IK, UNetDC_short_ECA_assp,\
        UNetDC_short_assp, UNetDC_short_first_assp, UNetDC_short_mid_assp, UNetDC_short_last_assp, \
        UNetDC_short_last_assp_ECA, UNetDC_ECA_ASSP_1, \
        UNetDC_ECA_ASSP_2, UNetDC_ECA_ASSP_3, UNetDC_ECA_ASSP_4, UNetDC_short_first_assp_ECA, \
        UNetDC_short_mid_assp_ECA, UNetDC_short_last_assp_ECA,UNetDC_halfchannel,\
        UNetDC_short_first_assp_allECA
from .unet_model_r import UNetDC_short_first_rassp_ECA, UNetDC_short_mid_rassp_ECA, UNetDC_short_last_rassp_ECA, \
        UNetDC_short_ECA_rassp,UNetDC_short_first_rassp, UNetDC_short_first_rassp_ECA_ScConv, \
            UNetDC_short_first_rassp_ECAfront_ScConv, UNetDC_short_first_rassp_first_ECA, \
                UNetDC_short_first_rassp_ECA_half
from .res_unet import ResUnet,ResUnet_DC,ResUnet_DCIK, ResUnet_DC_Normal
from .res_unet_plus import ResUnetPlusPlus, ResUnetPlusPlus_DCIK,ResUnetPlusPlus_DC, ResUnetPlusPlus_DC_att, ResUnetPlusPlus_DC_att_assp_normal, \
        ResUnetPlusPlus_DC_att_Bigkernel,ResUnetPlusPlus_DC_att_Medkernel, ResUnetPlusPlus_DC_att_assp, ResUnetPlusPlus_DC_att_asspsame, \
        CAU_NetV1,CAU_NetV2,CAU_NetV3,CAU_NetV0
from .cascade_net import CascadeMRI,CascadeMRIM
from .hypernet import hyperNet
from .miccan import MICCAN, MICCANlong

def get_model(name):
    if name == "ResUnetPlusPlus_DCIK3":
        return ResUnetPlusPlus_DCIK(6,2)
    if name == "ResUnetPlusPlus_DC3":
        return ResUnetPlusPlus_DC(6,2)
    if name == "ResUnetPlusPlus_DC_att3":
        return ResUnetPlusPlus_DC_att(6,2)
    if name == "ResUnetPlusPlus_DC_att_assp3":
        return ResUnetPlusPlus_DC_att_assp(6,2)
    if name == "ResUnetPlusPlus_DC_att_asspsame3":
        return ResUnetPlusPlus_DC_att_asspsame(6,2)
    if name == "ResUnetPlusPlus_DC_att_assp_normal3":
        return ResUnetPlusPlus_DC_att_assp_normal(6,2)
    if name == "ResUnet_DC3":
        return ResUnet_DC(6,2)
    if name == "ResUnet_DC_Normal3":
        return ResUnet_DC_Normal(6,2)
    if name == "ResUnet_DCIK3":
        return ResUnet_DCIK(6,2)
    if name == "UNetDC3":
        return UNetDC(6,2)
    if name == "UNetDC5":
        return UNetDC(10,2)
    if name == "UNetDC_short3":
        return UNetDC_short(6,2)
    if name == "UNetDC1":
        return UNetDC(2,2)
    if name == "UNetDC_halfchannel3":
        return UNetDC_halfchannel(6,2)
    if name == "UNet3":
        return UNet(6,2)
    if name == "UNetDC_ECA_ASSP_1":
        return UNetDC_ECA_ASSP_1(6,2)
    if name == "UNetDC_ECA_ASSP_2":
        return UNetDC_ECA_ASSP_2(6,2)
    if name == "UNetDC_ECA_ASSP_3":
        return UNetDC_ECA_ASSP_3(6,2)    
    if name == "UNetDC_ECA_ASSP_4":
        return UNetDC_ECA_ASSP_4(6,2)
    if name == "UNetDC_short_ECA3":
        return UNetDC_short_ECA(6,2)
    if name == "UNetDC_short_ECA_assp3":
        return UNetDC_short_ECA_assp(6,2)
    if name == "UNetDC_short_assp3":
        return UNetDC_short_assp(6,2)
    if name == "UNetDC_short_first_assp3":
        return UNetDC_short_first_assp(6,2)
    if name == "UNetDC_short_mid_assp3":
        return UNetDC_short_mid_assp(6,2)
    if name == "UNetDC_short_last_assp3":
        return UNetDC_short_last_assp(6,2)
    if name == "UNetDC_short_last_assp_ECA3":
        return UNetDC_short_last_assp_ECA(6,2)
    if name == "UNetDC_short_first_assp_ECA3":
        return UNetDC_short_first_assp_ECA(6,2)
    if name == "UNetDC_short_mid_assp_ECA3":
        return UNetDC_short_mid_assp_ECA(6,2)
    if name == "UNetDC_short_ECA_rassp3":
        return UNetDC_short_ECA_rassp(6,2)
    if name == "UNetDC_short_first_rassp_ECA5":
        return UNetDC_short_first_rassp_ECA(10,2)    
    if name == "UNetDC_short_first_rassp_ECA3":
        return UNetDC_short_first_rassp_ECA(6,2)
    if name == "UNetDC_short_first_rassp_ECA1":
        return UNetDC_short_first_rassp_ECA(2,2)    
    if name == "UNetDC_short_first_assp_allECA3":
        return UNetDC_short_first_assp_allECA(6,2)
    if name == "UNetDC_short_first_assp_allECA5":
        return UNetDC_short_first_assp_allECA(10,2)
    if name == "UNetDC_short_first_rassp_ECA_half3":
        return UNetDC_short_first_rassp_ECA_half(6,2)
    if name == "UNetDC_short_mid_rassp_ECA3":
        return UNetDC_short_mid_rassp_ECA(6,2)
    if name == "UNetDC_short_last_rassp_ECA3":
        return UNetDC_short_last_rassp_ECA(6,2)
    if name == "UNetDC_short_first_rassp3":
        return UNetDC_short_first_rassp(6,2)
    if name == "UNetDC_short_first_rassp_first_ECA3":
        return UNetDC_short_first_rassp_first_ECA(6,2)
    if name == "UNetDC_short_first_rassp_ECA_ScConv3":
        return UNetDC_short_first_rassp_ECA_ScConv(6,2)
    if name == "UNetDC_short_first_rassp_ECAfront_ScConv3":
        return UNetDC_short_first_rassp_ECAfront_ScConv(6,2)
    if name == "UNetDC_IK3":
        return UNetDC_IK(6,2)
    if name == "CascadeMRI3d5c5":
        return CascadeMRIM(6,2,5,5)
    if name == "CascadeMRI3d11c11":
        return CascadeMRIM(6,2,11,11)
    if name == "CascadeMRI1d5c5":
        return CascadeMRIM(2,2,5,5)
    if name == "CascadeMRI1d11c11":
        return CascadeMRIM(2,2,11,11)
    if name == "MICCAN1":
        print("MICCAN1 MICCAN1")
        return MICCAN(2,2)
    if name == "hyperNet3":
        return hyperNet(6,2)
    if name == "CAU_NetV1":
        return CAU_NetV1(6,2)
    if name == "CAU_NetV2":
        return CAU_NetV2(6,2)
    if name == "CAU_NetV3":
        return CAU_NetV3(6,2)
    if name == "CAU_NetV0":
        return CAU_NetV0(6,2)
    if name == "ResUnetPlusPlus_DC_att_Bigkernel3":
        return ResUnetPlusPlus_DC_att_Bigkernel(6,2)