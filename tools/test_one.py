from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.apis import show_result_pyplot
import os
import argparse
from mmcv import Config, DictAction
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('out', help='out file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    imagepath = r'/home/btr/belt_dataset/im_di/' #需要加载的测试图片的文件路径
    savepath = args.out #保存测试图片的路径
    config_file = Config.fromfile(args.config)
    device = 'cuda:0'
    checkpoint_file = args.checkpoint
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    # inference the demo image
    
    for filename in os.listdir(imagepath):
        img = os.path.join(imagepath, filename)
        result = inference_detector(model, img)
        out_file = os.path.join(savepath, filename)
        show_result_pyplot(model, img, result, score_thr=0.5, out_file=out_file)

if __name__ == '__main__':
    main()
