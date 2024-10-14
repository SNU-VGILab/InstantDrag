# InstantDrag

<p align="center">
  <img src="assets/demo.gif" alt="Demo video">
</p>

<br/>

Official implementation of the paper **"InstantDrag: Improving Interactivity in Drag-based Image Editing"** (SIGGRAPH Asia 2024).

<p align="center">
  <a href="https://arxiv.org/abs/2409.08857"><img src="https://img.shields.io/badge/arxiv-2409.08857-b31b1b"></a>
  <a href="https://joonghyuk.com/instantdrag-web/"><img src="https://img.shields.io/badge/Project%20Page-InstantDrag-blue"></a> 
  <a href="https://huggingface.co/alex4727/InstantDrag"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-forestgreen"></a>
</p>

---

## Setup

1. Create and activate a conda environment:
   ```bash
   conda create -n instantdrag python=3.10 -y
   conda activate instantdrag
   ```

2. Install PyTorch:
   ```bash
   pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
   ```

3. Install other dependencies:
   ```bash
   pip install transformers==4.44.2 diffusers==0.30.1 accelerate==0.33.0 gradio==4.44.0 opencv-python
   ```
   **Note:** Exact version matching may not be necessary for all dependencies.

## Demo

To run the demo:
```bash
cd demo/
CUDA_VISIBLE_DEVICES=0 python run_demo.py
```
### Disclaimer

- Our **base** models are **solely** trained on real-world talking head (facial) videos, with a focus on achieving **fast fine-grained facial editing w/o metadata**. The preliminary signs of generalizability in other types of scenes, without fine-tuning, should be considered more of an experimental byproduct and may not perform well in many cases. Please check the Appendix A of our paper for more information.
- This is a research project, **NOT** a commercial product. Use at your own risk.

### Usage Instructions & Tips

- Upload and preprocess image using Gradio's interface.
- Click to define source and target point pairs on the image.
- Adjust settings in the "Configs" tab.
  - We provide two checkpoints for FlowGen: config-2 (default, used for most figures in the paper) and config-3 (used for benchmark table in the paper). Generally, we recommend config-2 for most cases including few keypoints-based draggings. For extremely fine-grained editing with many drags (i.e. 68 keypoint drags as used in the benchmark), config-3 could be better suited as it produces more local movements.
  - If image moves too much or too little, try modifying the image or flow guidance scales (usually 1 ~ 2 are recommended, but flow guidance can be larger).
  - If you observe loss of identity or noisy artifacts, increasing image guidance or sampling steps could be helpful ([1.75, 1.5] scale is also a good choice for facial images).
- Click `Run` to perform the editing.
  - We recommend first viewing the example videos (in project page or .gif) and paper figures to understand the model's capabilities. Then, begin with facial images using fine-grained keypoint drags before progressing to more complex motions.
  - As noted in the paper, our model may struggle with large motions that exceed the capabilities of the optical flow estimation networks used for training data extraction.
- Notes on FlowGen Output Scale
  - In many cases, especially for unseen domains, FlowGen's output doesn't precisely span the -1 to 1 range expected by FlowDiffusion's fixed-size normalization process. For all figures and benchmarks in our paper, we applied a static multiplier of 2 based on observations to adjust FlowGen's output to match the expected range. However, we found that forcefully rescaling the output to -1 to 1 also works well, so we set this as the default behavior (when value is -1). While not recommended, you can manually modify this value to scale the output of FlowGen before feeding it to FlowDiffusion for larger or smaller motions.

**Note:** The initial run may take longer as models are loaded to GPU.

## BibTeX
If you find this work useful, please cite them as below!
```
@inproceedings{shin2024instantdrag,
      title     = {{InstantDrag: Improving Interactivity in Drag-based Image Editing}},
      author    = {Shin, Joonghyuk and Choi, Daehyeon and Park, Jaesik},
      booktitle = {ACM SIGGRAPH Asia 2024 Conference Proceedings},
      year      = {2024},
      pages     = {1--10},
}
```
