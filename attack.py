# Gaussian Splitting Attack

#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#


import os
import torch
from utils.loss_utils import l1_loss,l2_loss, ssim
from utils.image_utils import psnr
from adv_render import adv_render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from torchvision.models.inception import Inception3
from torchvision.models import resnet50, vit_b_16
from torchvision.models.vgg import vgg16
from torch import nn
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from torchvision.utils import save_image
from dist_utils import ChamferDist
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is4chan(tensor_img):
    channels = tensor_img.size()[0]
    if channels == 4:
        tensor_img_size = tensor_img.size()
        tensor_img_alpha = tensor_img[3, :, :].unsqueeze(0).broadcast_to([3, tensor_img_size[1], tensor_img_size[2]])
        tensor_img_rgb = tensor_img[:3, :, :]
        tensor_img_white = torch.ones_like(tensor_img_rgb) * 255
        tensor_img = torch.where(tensor_img_alpha > 0, tensor_img_rgb, tensor_img_white)
    return tensor_img
    
def test_render(model, views, gaussians, pipe, background, ori_image):
    running_corrects=0
    iter_num=0
    allpsnr = 0
    print('')
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    
        with torch.no_grad():
          with torch.set_grad_enabled(False):

            rendering = adv_render(view, gaussians, pipe, background)["render"]
            inputs_rend = rendering[[2,1,0],:,:].unsqueeze(0)*255              
            outputs = model(inputs_rend)
            _, preds = torch.max(outputs, 1)
            #####
            gt = ori_image[idx]#is4chan(view.original_image)
            inputs_gt = gt[[2,1,0],:,:].unsqueeze(0)*255 
            outputs_gt = model(inputs_gt)
            _, labels = torch.max(outputs_gt, 1)
            running_corrects += torch.sum(preds == labels)
            iter_num += preds.size(0)
            
            allpsnr += psnr(inputs_rend/255.0, inputs_gt/255.0).mean().double()
    print(preds,labels)
    print('Acc: {:.4f}--{}/{} viewers, psnr={}'.format(running_corrects/iter_num, running_corrects, iter_num, allpsnr/iter_num))
    return running_corrects/iter_num, allpsnr/iter_num

def save_img(view, gaussians, pipeline, background, ori_image, path, iteration, acc, psnr):
    with torch.no_grad():
        rendering = adv_render(view, gaussians, pipeline, background)["render"]
        gt = ori_image[0]
        save_image(rendering, os.path.join(path, 'r_{0:d}_{1:.3f}_{2:.2f}'.format(iteration,acc,psnr) + ".png"))
        save_image(gt, os.path.join(path, "gt.png"))

def get_adv_loss(outputs, labels, target=False, kappa=0):
    one_hot = torch.eye(len(outputs[0])).cuda()[labels]
    i,_ = torch.max((1-one_hot)*outputs,dim=1)
    j = torch.masked_select(outputs, one_hot.bool())
    if target:
        return torch.clamp(i-j,min=-kappa)
    else:
        return torch.clamp(j-i,min=-kappa)

def remove_duplicate_rows(exp_points):
    unique_points = set()
    unique_indices = []
    for idx, point in enumerate(exp_points):
        # Converting coordinate pairs to tuples for inclusion in collections
        point_tuple = (point[0].item(), point[1].item())
        if point_tuple not in unique_points:
            unique_points.add(point_tuple)
            unique_indices.append(idx)
    exp_points_unique = exp_points[unique_indices]
    return exp_points_unique 

def training(dataset, opt, pipe, checkpoint, args):
    class_list = {"chair": 0, "drums": 1, "ficus": 2, "hotdog": 3, "lego": 4, "materials": 5, "mic": 6, "ship": 7}
    num_classes = len(class_list.keys())
    # inception v3 model
    if args.model_name == "resnet50":
        model = resnet50(num_classes=num_classes)
        weights_path = Path("./model/weights/resnet50_8_best.pth")
    elif args.model_name == "vit_b_16":
        model = vit_b_16(num_classes=num_classes)
        weights_path = Path("./model/weights/vit_b_16_8_best.pth")
    pretrain_model = torch.load(weights_path.absolute(), map_location=device)
    model.load_state_dict(pretrain_model)
    model.to('cuda')
    model.eval()  # Set model to evaluate mode
    target_label = torch.tensor([args.target_class], device=device)
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Attacking progress")
    first_iter += 1
    max_acc = 1.0
    chamfer_loss = ChamferDist()

    print('lr:',str(opt.position_lr_init * gaussians.spatial_lr_scale),str(opt.feature_lr),str(opt.scaling_lr),str(opt.rotation_lr),str(opt.opacity_lr))
    ori_image = []
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        if iteration == first_iter:
            for viewpoint_cam in viewpoint_stack:
                bg = torch.rand((3), device=device) if opt.random_background else background
                render_pkg = adv_render(viewpoint_cam, gaussians, pipe, bg)
                image = render_pkg["render"]
                ori_image.append(image.detach())
                if args.select_rate > 0:
                    inputs_rend = image[[2,1,0],:,:].unsqueeze(0)*255
                    outputs = model(inputs_rend)
                    loss = F.cross_entropy(outputs, target_label.broadcast_to([outputs.size()[0], ]))
                    loss.backward()
            
            grad_all = torch.zeros((gaussians._xyz.size(0))).cuda()
            for param in [gaussians._xyz,gaussians._features_dc,gaussians._features_rest,
                        gaussians._scaling,gaussians._rotation,gaussians._opacity]:
                grad = torch.abs(param.grad.data)
                min_val = torch.min(grad)
                max_val = torch.max(grad)

                # Perform min-max normalization
                range_val = max_val - min_val
                normalized_grad = (grad - min_val) / range_val
                for _ in normalized_grad.size()[1:]:
                    normalized_grad = torch.mean(normalized_grad,1)
                grad_all += normalized_grad
            # Calculate the number of the first k elements
            percentile_count = int(gaussians._xyz[:,0].numel() * args.select_rate)
            selected_pts_mask = []
            while percentile_count>0:
                if percentile_count>gaussians._xyz[:,0].numel():
                    now_count = gaussians._xyz[:,0].numel()
                else:now_count = percentile_count
                _, mask_0 = torch.topk(grad_all.view(-1), k=now_count, largest=True)
                percentile_count = percentile_count - gaussians._xyz[:,0].numel()
                selected_pts_mask.append(mask_0)
            selected_pts_mask = torch.cat(selected_pts_mask)
            print(selected_pts_mask)

            # Copy point, initialize 3D perturbation
            extension_xyz0 = gaussians._xyz[selected_pts_mask].clone()
            extension_features_dc0 = gaussians._features_dc[selected_pts_mask].clone()
            extension_features_rest0 = gaussians._features_rest[selected_pts_mask].clone()
            scaling_min = torch.min(gaussians._scaling,0)
            extension_scaling0 = torch.tile(scaling_min.values.unsqueeze(0), (extension_xyz0.size(0),1)).clone()
            rotation_min = torch.min(gaussians._rotation,0)
            extension_rotation0 = torch.tile(rotation_min.values.unsqueeze(0), (extension_xyz0.size(0),1)).clone()
            extension_opacity0 = gaussians._opacity[selected_pts_mask].clone()

            print(extension_xyz0.size())
            ori_size = gaussians._xyz.size(0)
            print("init success!")
            # Cat
            gaussians._xyz.data = torch.cat((gaussians._xyz.data, extension_xyz0), dim=0)
            gaussians._features_dc.data = torch.cat((gaussians._features_dc.data, extension_features_dc0), dim=0)
            gaussians._features_rest.data = torch.cat((gaussians._features_rest.data, extension_features_rest0), dim=0)
            gaussians._scaling.data = torch.cat((gaussians._scaling.data, extension_scaling0), dim=0)
            gaussians._opacity.data = torch.cat((gaussians._opacity.data, extension_opacity0), dim=0)
            gaussians._rotation.data = torch.cat((gaussians._rotation.data, extension_rotation0), dim=0)
            
            # 
            gaussians_ori_xyz = gaussians._xyz.data
            gaussians_ori_features_dc= gaussians._features_dc.data
            gaussians_ori_features_rest = gaussians._features_rest.data
            gaussians_ori_scaling = gaussians._scaling.data
            gaussians_ori_opacity = gaussians._opacity.data
            gaussians_ori_rotation = gaussians._rotation.data
            
            # Initialize perturbation point parameters
            extension_xyz = nn.Parameter(torch.zeros_like(gaussians_ori_xyz)).requires_grad_(True).cuda()
            extension_features_dc = nn.Parameter(torch.zeros_like(gaussians_ori_features_dc)).requires_grad_(True).cuda()
            extension_features_rest = nn.Parameter(torch.zeros_like(gaussians_ori_features_rest)).requires_grad_(True).cuda()
            extension_scaling = nn.Parameter(torch.zeros_like(gaussians_ori_scaling)).requires_grad_(True).cuda()
            extension_opacity = nn.Parameter(torch.zeros_like(gaussians_ori_opacity)).requires_grad_(True).cuda()
            extension_rotation = nn.Parameter(torch.zeros_like(gaussians_ori_rotation)).requires_grad_(True).cuda()
            
            print(extension_xyz.size())
            continue

        # the original Gaussian does not add a perturbation
        with torch.no_grad():
            extension_xyz[:ori_size]=0
            extension_features_dc[:ori_size]=0
            extension_features_rest[:ori_size]=0
            extension_scaling[:ori_size]=0
            extension_opacity[:ori_size]=0
            extension_rotation[:ori_size]=0
        # Perturbation additions
        gaussians._xyz = gaussians_ori_xyz + extension_xyz
        gaussians._features_dc = gaussians_ori_features_dc + extension_features_dc
        gaussians._features_rest =gaussians_ori_features_rest + extension_features_rest
        gaussians._scaling = gaussians_ori_scaling + extension_scaling
        gaussians._opacity = gaussians_ori_opacity + extension_opacity
        gaussians._rotation = gaussians_ori_rotation + extension_rotation

        # iteration
        for ind,viewpoint_cam in enumerate(viewpoint_stack):
            bg = torch.rand((3), device=device) if opt.random_background else background
            render_pkg = adv_render(viewpoint_cam, gaussians, pipe, bg)
            image, means2D, points_contribute = render_pkg["render"], render_pkg["means2D"], render_pkg["points_contribute"]
            inputs_rend = image[[2,1,0],:,:].unsqueeze(0)*255
            outputs = model(inputs_rend)

            # adversarial loss
            adv_loss = F.cross_entropy(outputs, target_label.broadcast_to([outputs.size()[0], ]))

            # position loss
            dis_loss = chamfer_loss(gaussians._xyz, gaussians_ori_xyz)
            
            # rendering loss
            gt_image = ori_image[ind]
            Ll1 = l1_loss(image, gt_image)
            cons_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            # color loss
            exp_points = torch.round(means2D[ori_size:].detach()).to(dtype=torch.int32)
            not_0_index = points_contribute[ori_size:]!=0
            exp_points = exp_points[not_0_index]
            # Non-repeating pixel positions
            exp_points_flat = exp_points[:, 0] * 1000 + exp_points[:, 1]
            _, unique_indices = np.unique(exp_points_flat.cpu(), return_index=True)
            exp_points_unique = exp_points[unique_indices]
            x_coords = torch.clamp(exp_points_unique[:, 0],0,298)
            y_coords = torch.clamp(exp_points_unique[:, 1],0,298)
            # Index Pixel Value
            now_rgb = image[:, y_coords, x_coords]
            ori_rgb = gt_image[:, y_coords, x_coords]
            img_loss = l2_loss(now_rgb,ori_rgb)

            loss = args.lambda_adv*adv_loss+ args.lambda_cons*cons_loss + \
                   args.lambda_dis*dis_loss + args.lambda_col*img_loss
            loss.backward()

        change_xyz = extension_xyz.grad.data
        change_xyz = opt.position_lr_init * gaussians.spatial_lr_scale*change_xyz.sign()
        extension_xyz.data = (extension_xyz.data - change_xyz).detach()
        extension_xyz.grad.zero_()
        
        change_features_dc = extension_features_dc.grad.data           
        change_features_dc = opt.feature_lr*change_features_dc.sign()      
        extension_features_dc.data = (extension_features_dc.data - change_features_dc).detach()
        extension_features_dc.grad.zero_()
            
        change_features_rest = extension_features_rest.grad.data
        change_features_rest = opt.feature_lr*change_features_rest.sign()
        extension_features_rest.data = (extension_features_rest.data - change_features_rest).detach()
        extension_features_rest.grad.zero_()
            
        change_scaling = extension_scaling.grad.data
        change_scaling = opt.scaling_lr*change_scaling.sign()
        extension_scaling.data = (extension_scaling.data - change_scaling).detach()
        extension_scaling.grad.zero_()
            
        change_rotation = extension_rotation.grad.data
        change_rotation = opt.rotation_lr*change_rotation.sign()
        extension_rotation.data = (extension_rotation.data - change_rotation).detach()
        extension_rotation.grad.zero_()
            
        change_opacity = extension_opacity.grad.data
        change_opacity = opt.opacity_lr*change_opacity.sign()
        extension_opacity.data = (extension_opacity.data - change_opacity).detach()
        extension_opacity.grad.zero_()
            
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 1 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
                torch.cuda.empty_cache()
            if iteration == opt.iterations:
                progress_bar.close()
            if iteration % 10==0:
                attack_acc,allpsnr = test_render(model, scene.getTrainCameras(), gaussians, pipe, bg,ori_image)
                if max_acc - attack_acc >=0.2:
                    scene.save(iteration)
                    save_img(scene.getTrainCameras()[0], gaussians, pipe, bg, ori_image,
                             scene.model_path, iteration,attack_acc,allpsnr)
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    max_acc = attack_acc
                if iteration >= 51000:
                    scene.save(iteration)
                    save_img(scene.getTrainCameras()[0], gaussians, pipe, bg, ori_image, 
                             scene.model_path, iteration,attack_acc,allpsnr)
                    print("\n[ITER {}] Saving Gaussians".format(iteration),'attack success!')
                    break
            if iteration % 200 == 0:
                print(os.path.join(scene.model_path, 'r_{0:03d}'.format(iteration) + ".png"))
                save_img(scene.getTrainCameras()[0], gaussians, pipe, bg, ori_image,
                         scene.model_path, iteration,attack_acc,allpsnr)
                scene.save(iteration)
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                
def prepare_output_and_logger(args,args_all):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    
    print("Output folder: {}".format(args.model_path))
    # Set up output folder
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    from shutil import copyfile
    name=os.path.basename(__file__)
    copyfile(os.path.join(os.getcwd(), name), os.path.join(args.model_path, name))
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--evl_iter', type=int, default=5_00)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000,50_000,100_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000,50_000,100_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000,50_000,100_000])
    parser.add_argument("--start_checkpoint", type=str, default = './output/')

    parser.add_argument("--lambda_adv", type=float, default=1.0)
    parser.add_argument("--lambda_cons", type=float, default=100.0)
    parser.add_argument("--lambda_dis", type=float, default=1.0)
    parser.add_argument("--lambda_col", type=float, default=100.0)

    parser.add_argument("--select_rate", type=float, default=0.5)

    parser.add_argument("--source_gs", type=str, default='lego')
    parser.add_argument("--model_name", type=str, default='vit_b_16')
    parser.add_argument("--target_class", type=int, default=3)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.source_path="../../Data/nerf_synthetic/"
    # 初始化原高斯模型
    args.start_checkpoint+=args.source_gs
    args.source_path+=args.source_gs
    args.start_checkpoint+='299/chkpnt50000.pth'
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(args.resolution)
    # Start GUI server, configure and run training
    #network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint, args)

    # All done
    print("\nTraining complete.")
