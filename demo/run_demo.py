import os
import torch
import gradio as gr
from huggingface_hub import snapshot_download
os.makedirs("checkpoints", exist_ok=True)
snapshot_download("alex4727/InstantDrag", local_dir="./checkpoints")

from demo_utils import (
    process_img,
    get_points,
    undo_points_image,
    clear_all,
    InstantDragPipeline,
)

LENGTH = 480  # Length of the square area displaying/editing images

with gr.Blocks() as demo:
    pipeline = InstantDragPipeline(seed=42, device="cuda", dtype=torch.float16)

    # Layout definition
    with gr.Row():
        gr.Markdown(
            """
            # InstantDrag: Improving Interactivity in Drag-based Image Editing
            """
        )

    with gr.Tab(label="InstantDrag Demo"):
        selected_points = gr.State([])         # Store points
        original_image = gr.State(value=None)  # Store original input image

        with gr.Row():
            # Upload & Preprocess Image Column
            with gr.Column():
                gr.Markdown(
                    """<p style="text-align: center; font-size: 20px">Upload & Preprocess Image</p>"""
                )
                canvas = gr.ImageEditor(
                    height=LENGTH,
                    width=LENGTH,
                    type="numpy",
                    image_mode="RGB",
                    label="Preprocess Image",
                    show_label=True,
                    interactive=True,
                )
                with gr.Row():
                    save_results = gr.Checkbox(
                        value=False,
                        label="Save Results",
                        scale=1,
                    )
                    undo_button = gr.Button("Undo Clicked Points", scale=3)

            # Click Points Column
            with gr.Column():
                gr.Markdown(
                    """<p style="text-align: center; font-size: 20px">Click Points</p>"""
                )
                input_image = gr.Image(
                    type="numpy",
                    label="Click Points",
                    show_label=True,
                    height=LENGTH,
                    width=LENGTH,
                    interactive=False,
                    show_fullscreen_button=False,
                )
                with gr.Row():
                    run_button = gr.Button("Run")

            # Editing Results Column
            with gr.Column():
                gr.Markdown(
                    """<p style="text-align: center; font-size: 20px">Editing Results</p>"""
                )
                edited_image = gr.Image(
                    type="numpy",
                    label="Editing Results",
                    show_label=True,
                    height=LENGTH,
                    width=LENGTH,
                    interactive=False,
                    show_fullscreen_button=False,
                )
                with gr.Row():
                    clear_all_button = gr.Button("Clear All")

        with gr.Tab("Configs - make sure to check README for details"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        flowgen_choices = sorted(
                            [model for model in os.listdir("checkpoints/") if "flowgen" in model]
                        )
                        flowgen_ckpt = gr.Dropdown(
                            value=flowgen_choices[0],
                            label="Select FlowGen to use",
                            choices=flowgen_choices,
                            info="config2 for most cases, config3 for more fine-grained dragging",
                            scale=2,
                        )
                        flowdiffusion_choices = sorted(
                            [model for model in os.listdir("checkpoints/") if "flowdiffusion" in model]
                        )
                        flowdiffusion_ckpt = gr.Dropdown(
                            value=flowdiffusion_choices[0],
                            label="Select FlowDiffusion to use",
                            choices=flowdiffusion_choices,
                            info="single model for all cases",
                            scale=1,
                        )
                        image_guidance = gr.Number(
                            value=1.5,
                            label="Image Guidance Scale",
                            precision=2,
                            step=0.1,
                            scale=1,
                            info="typically between 1.0-2.0.",
                        )
                        flow_guidance = gr.Number(
                            value=1.5,
                            label="Flow Guidance Scale",
                            precision=2,
                            step=0.1,
                            scale=1,
                            info="typically between 1.0-5.0",
                        )
                        num_steps = gr.Number(
                            value=20,
                            label="Inference Steps",
                            precision=0,
                            step=1,
                            scale=1,
                            info="typically between 20-50, 20 is usually enough",
                        )
                        flowgen_output_scale = gr.Number(
                            value=-1.0,
                            label="FlowGen Output Scale",
                            precision=1,
                            step=0.1,
                            scale=2,
                            info="-1.0, by default, forces flowgen's output to [-1, 1], could be adjusted to [0, ∞] for stronger/weaker effects",
                        )

        gr.Markdown(
            """
            <p style="text-align: center; font-size: 18px;">Examples</p>
            """
        )
        with gr.Row():
            gr.Examples(
                examples=[
                    "samples/airplane.jpg",
                    "samples/anime.jpg",
                    "samples/caligraphy.jpg",
                    "samples/crocodile.jpg",
                    "samples/elephant.jpg",
                    "samples/meteor.jpg",
                    "samples/monalisa.jpg",
                    "samples/portrait.jpg",
                    "samples/sketch.jpg",
                    "samples/surreal.jpg",
                ],
                inputs=[canvas],
                outputs=[original_image, selected_points, input_image],
                fn=process_img,
                cache_examples=False,
                examples_per_page=10,
            )
        gr.Markdown(
            """
            <p style="text-align: center; font-size: 9">[Important] Our base models are solely trained on real-world talking head (facial) videos, with a focus on achieving fine-grained facial editing. <br>
            Their application to other types of scenes, without fine-tuning, should be considered more of an experimental byproduct and may not perform well in many cases (we currently support only square images).</p>
            """
        )

    # Event Handlers
    canvas.change(
        process_img,
        [canvas],
        [original_image, selected_points, input_image],
    )

    input_image.select(
        get_points,
        [input_image, selected_points],
        [input_image],
    )

    undo_button.click(
        undo_points_image,
        [original_image],
        [input_image, selected_points],
    )

    run_button.click(
        pipeline.run,
        [
            original_image,
            selected_points,
            flowgen_ckpt,
            flowdiffusion_ckpt,
            image_guidance,
            flow_guidance,
            flowgen_output_scale,
            num_steps,
            save_results,
        ],
        [edited_image],
    )

    clear_all_button.click(
        clear_all,
        [],
        [
            canvas,
            input_image,
            edited_image,
            selected_points,
            original_image,
        ],
    )

demo.queue().launch(share=False, debug=True)
