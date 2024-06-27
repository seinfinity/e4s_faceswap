import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import copy

from torch.nn import functional as F
from src.pretrained.face_vid2vid.driven_demo import init_facevid2vid_pretrained_model, drive_source_demo
from src.pretrained.gpen.gpen_demo import init_gpen_pretrained_model, GPEN_demo
from src.pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo, vis_parsing_maps
from src.utils.swap_face_mask import swap_head_mask_revisit_considerGlass
from src.utils import torch_utils
from src.utils.alignmengt import crop_faces, calc_alignment_coefficients
from src.utils.morphology import dilation, erosion
from src.utils.multi_band_blending import blending
from src.options.swap_options import SwapFacePipelineOptions
from src.models.networks import Net3
from src.datasets.dataset import TO_TENSOR, NORMALIZE, __celebAHQ_masks_to_faceParser_mask_detailed

def create_masks(mask, outer_dilation=0, operation='dilation'):
    radius = outer_dilation
    temp = copy.deepcopy(mask)
    if operation == 'dilation':
        full_mask = dilation(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device), engine='convolution')
        border_mask = full_mask - temp
    elif operation == 'erosion':
        full_mask = erosion(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device), engine='convolution')
        border_mask = temp - full_mask
    elif operation == 'expansion':
        full_mask = dilation(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device), engine='convolution')
        erosion_mask = erosion(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device), engine='convolution')
        border_mask = full_mask - erosion_mask

    border_mask = border_mask.clip(0, 1)
    content_mask = mask
    
    return content_mask, border_mask, full_mask 

def logical_or_reduce(*tensors):
    return torch.stack(tensors, dim=0).any(dim=0)

def logical_and_reduce(*tensors):
    return torch.stack(tensors, dim=0).all(dim=0)

def paste_image_mask(inverse_transform, image, dst_image, mask, radius=0, sigma=0.0):
    image_masked = image.copy().convert('RGBA')
    pasted_image = dst_image.copy().convert('RGBA')
    if radius != 0:
        mask_np = np.array(mask)
        kernel_size = (radius * 2 + 1, radius * 2 + 1)
        kernel = np.ones(kernel_size)
        eroded = cv2.erode(mask_np, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=255)
        blurred_mask = cv2.GaussianBlur(eroded, kernel_size, sigmaX=sigma)
        blurred_mask = Image.fromarray(blurred_mask)
        image_masked.putalpha(blurred_mask)
    else:
        image_masked.putalpha(mask)

    projected = image_masked.transform(dst_image.size, Image.PERSPECTIVE, inverse_transform, Image.BILINEAR)
    pasted_image.alpha_composite(projected)
    return pasted_image

def paste_image(coeffs, img, orig_image):
    pasted_image = orig_image.copy().convert('RGBA')
    projected = img.convert('RGBA').transform(orig_image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    pasted_image.paste(projected, (0, 0), mask=projected)
    return pasted_image

def smooth_face_boundary(image, dst_image, mask, radius=0, sigma=0.0):
    image_masked = image.copy().convert('RGBA')
    pasted_image = dst_image.copy().convert('RGBA')
    if radius != 0:
        mask_np = np.array(mask) 
        kernel_size = (radius * 2 + 1, radius * 2 + 1)
        kernel = np.ones(kernel_size)
        eroded = cv2.erode(mask_np, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=255)  
        blurred_mask = cv2.GaussianBlur(eroded, kernel_size, sigmaX=sigma)
        blurred_mask = Image.fromarray(blurred_mask)
        image_masked.putalpha(blurred_mask)
    else:
        image_masked.putalpha(mask)

    pasted_image.alpha_composite(image_masked)
    return pasted_image

def crop_and_align_face(target_files):
    image_size = 1024
    scale = 1.0
    center_sigma = 0
    xy_sigma = 0
    use_fa = False
    
    print('Aligning images')
    crops, orig_images, quads = crop_faces(image_size, target_files, scale, center_sigma=center_sigma, xy_sigma=xy_sigma, use_fa=use_fa)
    
    inv_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads
    ]
    
    return crops, orig_images, quads, inv_transforms

def swap_comp_style_vector(style_vectors1, style_vectors2, comp_indices=[], belowFace_interpolation=False):
    assert comp_indices is not None
    
    style_vectors = copy.deepcopy(style_vectors1)
    
    for comp_idx in comp_indices:
        style_vectors[:,comp_idx,:] =  style_vectors2[:,comp_idx,:]
        
    if torch.sum(style_vectors2[:,7,:]) == 0:
        style_vectors[:,7,:] = (style_vectors1[:,7,:] + style_vectors2[:,7,:]) / 2   
    
    if torch.sum(style_vectors2[:,9,:]) == 0:
        style_vectors[:,9,:] = style_vectors1[:,9,:] 
    
    if belowFace_interpolation:
        style_vectors[:,8,:] = (style_vectors1[:,8,:] + style_vectors2[:,8,:]) / 2
    
    return style_vectors

@torch.no_grad()
def process_frame(source_img, target_frame, faceParsing_model, net, opts, generator, kp_detector, he_estimator, estimate_jacobian, GPEN_model):
    S = Image.open(source_img).convert("RGB").resize((1024, 1024))
    T = Image.fromarray(cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)).resize((1024, 1024))
    
    S_256, T_256 = [cv2.resize(np.array(im) / 255.0, (256, 256)) for im in [S, T]]  # 256, [0, 1] range
    T_mask = faceParsing_demo(faceParsing_model, T, convert_to_seg12=True, model_name=opts.faceParser_name)
    
    predictions = drive_source_demo(S_256, [T_256], generator, kp_detector, he_estimator, estimate_jacobian)
    predictions = [(pred * 255).astype(np.uint8) for pred in predictions]
    
    drivens = [GPEN_demo(pred[:, :, ::-1], GPEN_model, aligned=False) for pred in predictions]
    D = Image.fromarray(drivens[0][:, :, ::-1])  # to PIL.Image
    
    D_mask = faceParsing_demo(faceParsing_model, D, convert_to_seg12=True, model_name=opts.faceParser_name)

    driven = transforms.Compose([TO_TENSOR, NORMALIZE])(D)
    driven = driven.to(opts.device).float().unsqueeze(0)
    driven_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(D_mask))
    driven_mask = (driven_mask * 255).long().to(opts.device).unsqueeze(0)
    driven_onehot = torch_utils.labelMap2OneHot(driven_mask, num_cls=opts.num_seg_cls)

    target = transforms.Compose([TO_TENSOR, NORMALIZE])(T)
    target = target.to(opts.device).float().unsqueeze(0)
    target_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(T_mask))
    target_mask = (target_mask * 255).long().to(opts.device).unsqueeze(0)
    target_onehot = torch_utils.labelMap2OneHot(target_mask, num_cls=opts.num_seg_cls)
    
    driven_style_vector, _ = net.get_style_vectors(driven, driven_onehot) 
    target_style_vector, _ = net.get_style_vectors(target, target_onehot)
    
    swapped_msk, hole_map = swap_head_mask_revisit_considerGlass(D_mask, T_mask)
    
    comp_indices = set(range(opts.num_seg_cls)) - {0, 4, 11, 10}
    swapped_style_vectors = swap_comp_style_vector(target_style_vector, driven_style_vector, list(comp_indices), belowFace_interpolation=False)
    
    swapped_msk = Image.fromarray(swapped_msk).convert('L')
    swapped_msk = transforms.Compose([TO_TENSOR])(swapped_msk)
    swapped_msk = (swapped_msk * 255).long().to(opts.device).unsqueeze(0)
    swapped_onehot = torch_utils.labelMap2OneHot(swapped_msk, num_cls=opts.num_seg_cls)
    
    swapped_style_codes = net.cal_style_codes(swapped_style_vectors)
    swapped_face, _, structure_feats = net.gen_img(torch.zeros(1, 512, 32, 32).to(opts.device), swapped_style_codes, swapped_onehot)
    swapped_face_image = torch_utils.tensor2im(swapped_face[0])
    
    outer_dilation = 5  
    mask_bg = logical_or_reduce(*[swapped_msk == clz for clz in [0, 11, 4]])
    is_foreground = torch.logical_not(mask_bg)
    hole_index = hole_map[None][None] == 255
    is_foreground[hole_index[None]] = True
    foreground_mask = is_foreground.float()
    
    content_mask, border_mask, full_mask = create_masks(foreground_mask, outer_dilation=outer_dilation)
    
    content_mask = F.interpolate(content_mask, (1024, 1024), mode='bilinear', align_corners=False)
    content_mask_image = Image.fromarray(255 * content_mask[0, 0, :, :].cpu().numpy().astype(np.uint8))
    full_mask = F.interpolate(full_mask, (1024, 1024), mode='bilinear', align_corners=False)
    full_mask_image = Image.fromarray(255 * full_mask[0, 0, :, :].cpu().numpy().astype(np.uint8))

    swapped_and_pasted = smooth_face_boundary(swapped_face_image, T, full_mask_image, radius=outer_dilation)
    
    return cv2.cvtColor(np.array(swapped_and_pasted.convert('RGB')), cv2.COLOR_RGB2BGR)

def video_to_frames(video_path):
    frames = []
    video_capture = cv2.VideoCapture(video_path)
    success, frame = video_capture.read()
    while success:
        frames.append(frame)
        success, frame = video_capture.read()
    video_capture.release()
    return frames

def frames_to_video(frames, output_path, fps):
    height, width, layers = frames[0].shape
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()

if __name__ == "__main__":
    opts = SwapFacePipelineOptions().parse()
    face_vid2vid_cfg = "./pretrained_ckpts/facevid2vid/vox-256.yaml"
    face_vid2vid_ckpt = "./pretrained_ckpts/facevid2vid/00000189-checkpoint.pth.tar"
    generator, kp_detector, he_estimator, estimate_jacobian = init_facevid2vid_pretrained_model(face_vid2vid_cfg, face_vid2vid_ckpt)
    
    gpen_model_params = {
        "base_dir": "./pretrained_ckpts/gpen/",
        "in_size": 512,
        "model": "GPEN-BFR-512",
        "use_sr": True,
        "sr_model": "realesrnet",
        "sr_scale": 4,
        "channel_multiplier": 2,
        "narrow": 1,
    }
    GPEN_model = init_gpen_pretrained_model(model_params=gpen_model_params)

    if opts.faceParser_name == "default":
        faceParser_ckpt = "./pretrained_ckpts/face_parsing/79999_iter.pth"
        config_path = ""
    elif opts.faceParser_name == "segnext":
        faceParser_ckpt = "./pretrained_ckpts/face_parsing/segnext.small.best_mIoU_iter_140000.pth"
        config_path = "./pretrained_ckpts/face_parsing/segnext.small.512x512.celebamaskhq.160k.py"
    else:
        raise NotImplementedError("Please choose a valid face parser, the current supported models are [default | segnext], but %s is given." % opts.faceParser_name)
        
    faceParsing_model = init_faceParsing_pretrained_model(opts.faceParser_name, faceParser_ckpt, config_path)
    print("Load pre-trained face parsing models success!")

    net = Net3(opts)
    net = net.to(opts.device)
    save_dict = torch.load(opts.checkpoint_path)
    net.load_state_dict(torch_utils.remove_module_prefix(save_dict["state_dict"], prefix="module."))
    net.latent_avg = save_dict['latent_avg'].to(opts.device)
    
    source_img = "path_to_source_image"
    target_video = "path_to_target_video.mp4"
    output_video = "path_to_output_video.mp4"
    
    frames = video_to_frames(target_video)
    fps = 30  # Adjust according to your video FPS

    processed_frames = []
    for frame in tqdm(frames, desc="Processing frames"):
        processed_frame = process_frame(source_img, frame, faceParsing_model, net, opts, generator, kp_detector, he_estimator, estimate_jacobian, GPEN_model)
        processed_frames.append(processed_frame)
    
    frames_to_video(processed_frames, output_video, fps)
