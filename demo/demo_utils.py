import sys
sys.path.append("../")

import os
import re
import time
import datetime
from copy import deepcopy

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import load_file

from utils.flow_utils import flow_to_image, resize_flow
from flowgen.models import UnetGenerator
from flowdiffusion.pipeline import FlowDiffusionPipeline

LENGTH = 512
FLOWGAN_RESOLUTION = [256, 256] # HxW
FLOWDIFFUSION_RESOLUTION = [512, 512] # HxW

def process_img(image):
    if image["composite"] is not None and not np.all(image["composite"] == 0):
        original_image = Image.fromarray(image["composite"]).resize((LENGTH, LENGTH), Image.BICUBIC)
        original_image = np.array(exif_transpose(original_image))
        return original_image, [], gr.Image(value=deepcopy(original_image), interactive=False)
    else:
        return (
            gr.Image(value=None, interactive=False),
            [],
            gr.Image(value=None, interactive=False),
        )

def get_points(img, sel_pix, evt: gr.SelectData):
    sel_pix.append(evt.index)
    print(sel_pix)
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            cv2.circle(img, tuple(point), 4, (255, 0, 0), -1)
        else:
            cv2.circle(img, tuple(point), 4, (0, 0, 255), -1)
        points.append(tuple(point))
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 2, tipLength=0.5)
            points = []
    img = img if isinstance(img, np.ndarray) else np.array(img)
    return img

def display_points(img, predefined_points, save_results):
    if predefined_points != "":
        predefined_points = predefined_points.split()
        predefined_points = [int(re.sub(r'[^0-9]', '', point)) for point in predefined_points]
        processed_points = []
        for i, point in enumerate(predefined_points):
            if i % 2 == 0:
                processed_points.append([point, predefined_points[i+1]])
        selected_points = processed_points

    print(selected_points)
    points = []
    for idx, point in enumerate(selected_points):
        if idx % 2 == 0:
            cv2.circle(img, tuple(point), 4, (255, 0, 0), -1)
        else:
            cv2.circle(img, tuple(point), 4, (0, 0, 255), -1)
        points.append(tuple(point))
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 2, tipLength=0.5)
            points = []
    img = img if isinstance(img, np.ndarray) else np.array(img)

    if save_results:
        if not os.path.isdir("results/drag_inst_viz"):
            os.makedirs("results/drag_inst_viz")
        save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
        to_save_img = Image.fromarray(img)
        to_save_img.save(f"results/drag_inst_viz/{save_prefix}.png")

    return img

def undo_points_image(original_image):
    if original_image is not None:
        return original_image, []
    else:
        return gr.Image(value=None, interactive=False), []

def clear_all():
    return (
        gr.Image(value=None, interactive=True),
        gr.Image(value=None, interactive=False),
        gr.Image(value=None, interactive=False),
        [],
        None
    )

class InstantDragPipeline:
    def __init__(self, seed=9999, device="cuda", dtype=torch.float16):
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.flowgen_ckpt, self.flowdiffusion_ckpt = None, None
        self.model_config = dict()

    def build_model(self):
        print("Building model...")
        if self.flowgen_ckpt != self.model_config["flowgen_ckpt"]:
            self.flowgen = UnetGenerator(input_nc=5, output_nc=2)
            self.flowgen.load_state_dict(
                load_file(os.path.join("checkpoints/", self.model_config["flowgen_ckpt"]), device="cpu")
            )
            self.flowgen.to(self.device)
            self.flowgen.eval()
            self.flowgen_ckpt = self.model_config["flowgen_ckpt"]

        if self.flowdiffusion_ckpt != self.model_config["flowdiffusion_ckpt"]:
            self.flowdiffusion = FlowDiffusionPipeline.from_pretrained(
                os.path.join("checkpoints/", self.model_config["flowdiffusion_ckpt"]),
                torch_dtype=self.dtype,
                safety_checker=None
            )
            self.flowdiffusion.to(self.device)
            self.flowdiffusion_ckpt = self.model_config["flowdiffusion_ckpt"]

    def drag(self, original_image, selected_points, save_results):
        scale = self.model_config["flowgen_output_scale"]
        original_image = torch.tensor(original_image).permute(2, 0, 1).unsqueeze(0).float()  # 1, 3, 512, 512
        original_image = 2 * (original_image / 255.) - 1  # Normalize to [-1, 1]
        original_image = original_image.to(self.device)

        source_points = []
        target_points = []
        for idx, point in enumerate(selected_points):
            cur_point = torch.tensor([point[0], point[1]])  # x, y 
            if idx % 2 == 0:
                source_points.append(cur_point)
            else:
                target_points.append(cur_point)

        torch.cuda.synchronize()
        start_time = time.time()

        # Generate sparse flow vectors
        point_vector_map = torch.zeros((1, 2, LENGTH, LENGTH))
        for source_point, target_point in zip(source_points, target_points):
            cur_x, cur_y = source_point[0], source_point[1]
            target_x, target_y = target_point[0], target_point[1]
            vec_x = target_x - cur_x
            vec_y = target_y - cur_y
            point_vector_map[0, 0, int(cur_y), int(cur_x)] = vec_x
            point_vector_map[0, 1, int(cur_y), int(cur_x)] = vec_y
        point_vector_map = point_vector_map.to(self.device)

        # Sample-wise normalize the flow vectors
        factor_x = torch.amax(torch.abs(point_vector_map[:, 0, :, :]), dim=(1, 2)).view(-1, 1, 1).to(self.device)
        factor_y = torch.amax(torch.abs(point_vector_map[:, 1, :, :]), dim=(1, 2)).view(-1, 1, 1).to(self.device)
        if factor_x >= 1e-8: # Avoid division by zero
            point_vector_map[:, 0, :, :] /= factor_x
        if factor_y >= 1e-8: # Avoid division by zero
            point_vector_map[:, 1, :, :] /= factor_y

        with torch.inference_mode():
            gan_input_image = F.interpolate(original_image, size=FLOWGAN_RESOLUTION, mode="bicubic") # 256 x 256
            point_vector_map = F.interpolate(point_vector_map, size=FLOWGAN_RESOLUTION, mode="bicubic") # 256 x 256
            gan_input = torch.cat([gan_input_image, point_vector_map], dim=1)
            flow = self.flowgen(gan_input) # -1 ~ 1

            if scale == -1.0:
                flow[:, 0, :, :] *= 1.0 / torch.amax(torch.abs(flow[:, 0, :, :]), dim=(1, 2)).view(-1, 1, 1) # force the range to be [-1 ~ 1]
                flow[:, 1, :, :] *= 1.0 / torch.amax(torch.abs(flow[:, 1, :, :]), dim=(1, 2)).view(-1, 1, 1) # force the range to be [-1 ~ 1]
            else:
                flow[:, 0, :, :] *= scale # manually adjust the scale
                flow[:, 1, :, :] *= scale # manually adjust the scale

            if factor_x >= 1e-8:
                flow[:, 0, :, :] *= factor_x * (FLOWGAN_RESOLUTION[1] / original_image.shape[3]) # width
            else:
                flow[:, 0, :, :] *= 0
            if factor_y >= 1e-8:
                flow[:, 1, :, :] *= factor_y * (FLOWGAN_RESOLUTION[0] / original_image.shape[2]) # height
            else:
                flow[:, 1, :, :] *= 0
            
            resized_flow = resize_flow(flow, (FLOWDIFFUSION_RESOLUTION[0]//8, FLOWDIFFUSION_RESOLUTION[1]//8), scale_type="normalize_fixed")

            kwargs = {
                "image": original_image.to(self.dtype),
                "flow": resized_flow.to(self.dtype),
                "num_inference_steps": self.model_config['n_inference_step'],
                "image_guidance_scale": self.model_config['image_guidance'],
                "flow_guidance_scale": self.model_config['flow_guidance'],
                "generator": self.generator,
            }
            edited_image = self.flowdiffusion(**kwargs).images[0]

        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Inference Time: {inference_time} seconds")

        if save_results:
            save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
            if not os.path.isdir("results/flows"):
                os.makedirs("results/flows")
            np.save(f"results/flows/{save_prefix}.npy", flow[0].detach().cpu().numpy())
            if not os.path.isdir("results/flow_visualized"):
                os.makedirs("results/flow_visualized")
            flow_to_image(flow[0].detach()).save(f"results/flow_visualized/{save_prefix}.png")
            if not os.path.isdir("results/edited_images"):
                os.makedirs("results/edited_images")
            edited_image.save(f"results/edited_images/{save_prefix}.png")
            if not os.path.isdir("results/drag_instructions"):
                os.makedirs("results/drag_instructions")
            with open(f"results/drag_instructions/{save_prefix}.txt", "w") as f:
                f.write(str(selected_points))

        edited_image = np.array(edited_image)
        return edited_image           

    def run(self, original_image, selected_points, 
            flowgen_ckpt, flowdiffusion_ckpt, image_guidance, flow_guidance, flowgen_output_scale,
            num_steps, save_results):

        self.model_config = {
            "flowgen_ckpt": flowgen_ckpt,
            "flowdiffusion_ckpt": flowdiffusion_ckpt, 
            "image_guidance": image_guidance,
            "flow_guidance": flow_guidance,
            "flowgen_output_scale": flowgen_output_scale,
            "n_inference_step": num_steps
        }

        self.build_model()

        edited_image = self.drag(original_image, selected_points, save_results)

        return edited_image