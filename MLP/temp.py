import os
import sys
import argparse
import torch
from pytorch_nndct.apis import torch_quantizer, Inspector
#
# from models.superglue_conv1d_trans import SuperGlue
from MLP import MLPNet

DIVIDER = '-----------------------------------------'


def quantize(build_dir, quant_mode, inspect):
    dset_dir = build_dir + '/dataset'
    float_model = build_dir + '/float_model'
    quant_model = build_dir + '/quant_model'
    inspect_result = build_dir + '/inspect_result'

    # use GPU if available
    if (torch.cuda.device_count() > 0):
        print('You have', torch.cuda.device_count(), 'CUDA devices available')
        for i in range(torch.cuda.device_count()):
            print(' Device', str(i), ': ', torch.cuda.get_device_name(i))
        print('Selecting device 0..')
        device = torch.device('cuda:0')
    else:
        print('No CUDA devices available..selecting CPU')
        device = torch.device('cpu')
    # load trained model
    model = MLPNet().eval().to(device)
    #model.load_state_dict(torch.load(os.path.join('./models/weights/superglue_indoor.pth')))

    # override batchsize if in test mode
    # if (quant_mode == 'test'):
    #     batchsize = 1

    # batchsize = 1 
    # descriptor_dim = 256
    # num_keypoints = 100  
    # H, W = 480, 640  

    # image0_rand = torch.randn(batchsize, 1, H, W)  # image0
    # image1_rand = torch.randn(batchsize, 1, H, W)  # image1
    # keypoints0_rand = torch.randn(batchsize, num_keypoints, 2)  # keypoints0
    # scores0_rand = torch.randn(batchsize, num_keypoints)  # scores0
    # descriptors0_rand = torch.randn(batchsize, descriptor_dim, num_keypoints)  # descriptors0
    # keypoints1_rand = torch.randn(batchsize, num_keypoints, 2)  # keypoints1
    # scores1_rand = torch.randn(batchsize, num_keypoints)  # scores1
    # descriptors1_rand = torch.randn(batchsize, descriptor_dim, num_keypoints)  # descriptors1

    # input = (image0_rand, image1_rand, keypoints0_rand, scores0_rand, descriptors0_rand, keypoints1_rand, scores1_rand, descriptors1_rand)

    vec = torch.randn(2796, 128)
    if quant_mode == 'float':
        quant_model = model
        if inspect:
            target = "DPUCZDX8G_ISA1_B4096"
            inspector = Inspector(target)
            inspector.inspect(quant_model, (vec,), device=device, output_dir=inspect_result, image_format="png")
            sys.exit()

    quantizer = torch_quantizer(quant_mode, model, input, output_dir=quant_model)
    quantized_model = quantizer.quant_model

    # if quant_mode == 'calib':
    #     quantizer.fast_finetune(test, (quantized_model))

    # evaluate
    quantized_model(vec).to(device)

    # export config
    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if quant_mode == 'test':
        quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)

    return


def run_main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir', type=str, default='build', help='Path to build folder. Default is build')
    ap.add_argument('-q', '--quant_mode', type=str, default='float', choices=['calib', 'test','float'],
                    help='Quantization mode (calib or test). Default is calib')
    ap.add_argument('--inspect', type=int, default=1 ,help='inspect model')
    args = ap.parse_args()

    print('\n' + DIVIDER)
    print('PyTorch version : ', torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print('--build_dir    : ', args.build_dir)
    print('--quant_mode   : ', args.quant_mode)
    print('--inspect    : ', args.inspect)
    print(DIVIDER)

    quantize(args.build_dir, args.quant_mode, args.inspect)

    return


if __name__ == '__main__':
    run_main()
