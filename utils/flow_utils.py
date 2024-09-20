import numpy as np
from PIL import Image
import torch 
import torch.nn.functional as F

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False, max_flow=None):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (torch.Tensor): Flow UV image of shape [2,H,W]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        PIL Image: Flow visualization image
    """
    flow_uv = flow_uv.permute(1, 2, 0).cpu().numpy() # change to [H,W,2] and convert to numpy

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    if max_flow is None:
        rad = np.sqrt(np.square(u) + np.square(v))
        rad_max = np.max(rad)
    else:
        rad_max = max_flow
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    flow_image = flow_uv_to_colors(u, v, convert_to_bgr)

    return Image.fromarray(flow_image)

def resize_flow(flow, size, scale_type="none", mode="bicubic"):
    """
    Resize the flow tensor (Bx2xHxW) to the given size (HxW).
    flow tensor is in range of [-ori_w, ori_w] and [-ori_h, ori_h]
    Size should be a tuple (H, W).
    """
    ori_h, ori_w = flow.shape[2:]
    flow = F.interpolate(flow, size=size, mode=mode, align_corners=False)

    if scale_type == "scale" and (ori_h != size[0] or ori_w != size[1]):
        flow[:,0,:,:] *= size[1] / ori_w
        flow[:,1,:,:] *= size[0] / ori_h
    elif scale_type == "normalize_fixed": # normalize to -1 ~ 1
        flow[:,0,:,:] /= ori_w
        flow[:,1,:,:] /= ori_h
    elif scale_type == "normalize_max":
        max_flow_x = torch.amax(torch.abs(flow[:, 0, :, :]), dim=(1, 2))
        max_flow_y = torch.amax(torch.abs(flow[:, 1, :, :]), dim=(1, 2))
        flow[:, 0, :, :] /= max_flow_x.view(-1, 1, 1)
        flow[:, 1, :, :] /= max_flow_y.view(-1, 1, 1)
    return flow