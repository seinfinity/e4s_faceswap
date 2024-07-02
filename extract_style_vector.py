import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from argparse import ArgumentParser
from skimage.transform import resize
from src.utils import torch_utils
from src.datasets.dataset import TO_TENSOR, NORMALIZE
from src.pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo
from src.models.networks import Net3
from src.options.swap_options import SwapFacePipelineOptions

def extract_and_save_style_vector(image_path, faceParsing_model, net, device, style_vector_path):
    image = Image.open(image_path).convert("RGB").resize((1024, 1024))
    image_256 = resize(np.array(image) / 255.0, (256, 256))
    mask = faceParsing_demo(faceParsing_model, image, convert_to_seg12=True)
    
    image_tensor = transforms.Compose([TO_TENSOR, NORMALIZE])(image).to(device).float().unsqueeze(0)
    mask_tensor = transforms.Compose([TO_TENSOR])(Image.fromarray(mask)).to(device).long().unsqueeze(0)
    onehot_mask = torch_utils.labelMap2OneHot(mask_tensor, num_cls=12)
    
    style_vector, _ = net.get_style_vectors(image_tensor, onehot_mask)
    torch.save(style_vector, style_vector_path)
    print(f"Style vector saved to {style_vector_path}")

if __name__ == "__main__":
    # Face parser 모델 초기화
    faceParsing_model = init_faceParsing_pretrained_model(opts.faceParser_name, opts.faceParser_ckpt, opts.config_path)
    
    # E4S 모델 초기화
    net = Net3(SwapFacePipelineOptions()).to(device)
    save_dict = torch.load(opts.checkpoint_path)
    net.load_state_dict(torch_utils.remove_module_prefix(save_dict["state_dict"], prefix="module."))
    net.latent_avg = save_dict['latent_avg'].to(device)

    # 스타일 벡터 추출 및 저장
    extract_and_save_style_vector(opts.source, faceParsing_model, net, device, opts.style_vector_path)
