import gradio as gr
import random
from S2I import Sketch2ImageController, css, scripts


class Sketch2ImageLaunch(Sketch2ImageController):
    def __init__(self):
        super().__init__(gr)

    def launch(self):
        with gr.Blocks(css=css) as demo:
            segment = gr.State(None)
            origin_frame = gr.State(None)
            click_stack = gr.State([[], []])  # Storage clicks status
            line = gr.Checkbox(label="line", value=False, elem_id="cb-line")
            eraser = gr.Checkbox(label="eraser", value=False, elem_id="cb-eraser")
            with gr.Row(elem_id="main_row"):
                with gr.Column(elem_id='column_segment'):
                    gr.Markdown("## SEGMENT", elem_id="segment_header")
                    segment_image = gr.Image(
                        source="upload",
                        type="pil",
                        image_mode="RGB",
                        invert_colors=True,
                        shape=(512, 512),
                        brush_radius=4,
                        height=440,
                        width=440,
                        brush_color="#000000",
                        interactive=True,
                        show_download_button=True,
                        elem_id="segment_image",
                        show_label=False,
                    )
                    run_button = gr.Button("Generate", min_width=50)
                    prompt = gr.Textbox(label="Prompt", value="", show_label=True)
                    with gr.Row():
                        style = gr.Dropdown(
                            label="Style",
                            choices=self.STYLE_NAMES,
                            value=self.DEFAULT_STYLE_NAME,
                            scale=1,
                        )
                        prompt_temp = gr.Textbox(
                            label="Prompt Style Template",
                            value=self.styles[self.DEFAULT_STYLE_NAME],
                            scale=2,
                            max_lines=1,
                        )
                        # with gr.Row():
                        val_r = gr.Slider(
                            label="Sketch guidance: ",
                            show_label=True,
                            minimum=0,
                            maximum=1,
                            value=0.4,
                            step=0.01,
                            scale=3,
                        )
                        # with gr.Row():
                        seed = gr.Textbox(label="Seed", value=42, scale=1, min_width=50)
                        randomize_seed = gr.Button(value='\U0001F3B2')
                with gr.Column(elem_id="column_input"):
                    gr.Markdown("## INPUT", elem_id="input_header")
                    image = gr.Image(
                        source="canvas",
                        tool="color-sketch",
                        type="pil",
                        image_mode="L",
                        invert_colors=True,
                        shape=(512, 512),
                        brush_radius=4,
                        height=440,
                        width=440,
                        brush_color="#000000",
                        interactive=True,
                        show_download_button=True,
                        elem_id="input_image",
                        show_label=False,
                    )
                    download_sketch = gr.Button(
                        "Download sketch", scale=1, elem_id="download_sketch"
                    )
                    gr.HTML(
                        """
                    <div class="button-row">
                        <div id="my-div-pencil" class="pad2"> <button id="my-toggle-pencil" onclick="return togglePencil(this)"></button> </div>
                        <div id="my-div-eraser" class="pad2"> <button id="my-toggle-eraser" onclick="return toggleEraser(this)"></button> </div>
                        <div class="pad2"> <button id="my-button-undo" onclick="return UNDO_SKETCH_FUNCTION(this)"></button> </div>
                        <div class="pad2"> <button id="my-button-clear" onclick="return DELETE_SKETCH_FUNCTION(this)"></button> </div>
                        <div class="pad2"> <button href="TODO" download="image" id="my-button-down" onclick='return theSketchDownloadFunction()'></button> </div>
                    </div>
                    """
                    )
                    tab_segment = gr.Tab(label="Segment Anything Setting", open=True)
                    with tab_segment:
                        with gr.Column():
                            models_download = gr.Textbox(label='Models Download Status')
                            model_type = gr.Radio(
                                choices=["vit_b", "vit_l", "vit_h"],
                                value="vit_b",
                                label="SAM Models Type",
                                interactive=True
                            )
                        with gr.Column():
                            point_mode = gr.Radio(
                                choices=["Positive", "Negative"],
                                value="Positive",
                                label="Point Prompt",
                                interactive=True)
                            click_undo_but = gr.Button(
                                value="Undo",
                                interactive=True
                            )

                with gr.Column(elem_id="column_output"):
                    gr.Markdown("## OUTPUT", elem_id="output_header")
                    result = gr.Image(
                        label="Result",
                        height=440,
                        width=440,
                        elem_id="output_image",
                        show_label=False,
                        show_download_button=True,
                    )
                    tab_image_input = gr.Tab(label="Upload Image")
                    with tab_image_input:
                        input_image = gr.File(label='Input image')
                    with gr.Accordion("SAM Advanced Options", open=True):
                        with gr.Row():
                            point_per_side = gr.Slider(label="point per sides", minimum=1, maximum=100,
                                                       value=16, step=1)
                            predict_iou_thresh = gr.Slider(label="IoU Threshold", minimum=0, maximum=1,
                                                           value=0.8,
                                                           step=0.1)
                            score_thresh = gr.Slider(label="Scored Threshold", minimum=0, maximum=1, value=0.9,
                                                     step=0.1)
                            crop_n_layers = gr.Slider(label="Crop Layers", minimum=0, maximum=100, value=1,
                                                      step=1)
                            crop_n_points = gr.Slider(label="Crop Points", minimum=0, maximum=100, value=2,
                                                      step=1)
                            min_mask_region_area = gr.Slider(label="Mask Region Area", minimum=0, maximum=1000,
                                                             value=100, step=100)

                download_output = gr.Button("Download output", elem_id="download_output")
            # inputs_sam = [segment,
            #               input_image,
            #               point_per_side,
            #               input_image,
            #               model_type,
            #               predict_iou_thresh,
            #               score_thresh,
            #               crop_n_layers,
            #               crop_n_points,
            #               min_mask_region_area]
            # outputs_sam = [segment_image, image, origin_frame]
            input_image.change(
                fn=self.get_meta_from_image,
                inputs=[input_image],
                outputs=[segment_image, origin_frame]
            )
            segment_image.select(
                fn=self.sam_click,
                inputs=[
                    segment, origin_frame, model_type, point_mode, click_stack,
                    point_per_side, predict_iou_thresh, score_thresh,
                    crop_n_layers, crop_n_points, min_mask_region_area
                ],
                outputs=[
                    segment, segment_image, click_stack, models_download
                ]
            )

            click_undo_but.click(
                fn=self.undo_click_stack_and_refine_seg,
                inputs=[
                    segment, origin_frame, click_stack
                ],
                outputs=[
                    segment, segment_image, click_stack
                ]
            )

            eraser.change(
                fn=lambda x: gr.update(value=not x),
                inputs=[eraser],
                outputs=[line],
                queue=False,
                api_name=False,
            ).then(self.update_canvas, [line, eraser], [image])
            line.change(
                fn=lambda x: gr.update(value=not x),
                inputs=[line],
                outputs=[eraser],
                queue=False,
                api_name=False,
            ).then(self.update_canvas, [line, eraser], [image])
            demo.load(None, None, None, _js=scripts)
            randomize_seed.click(
                lambda x: random.randint(0, self.MAX_SEED),
                inputs=[],
                outputs=seed,
                queue=False,
                api_name=False,
            )
            inputs = [image, prompt, prompt_temp, style, seed, val_r, segment_image, segment, origin_frame, model_type]
            outputs = [result, segment_image, download_sketch, download_output]
            prompt.submit(fn=self.artwork, inputs=inputs, outputs=outputs, api_name=False)
            style.change(
                lambda x: self.styles[x],
                inputs=[style],
                outputs=[prompt_temp],
                queue=False,
                api_name=False,
            ).then(
                fn=self.artwork,
                inputs=inputs,
                outputs=outputs,
                api_name=False,
            )
            val_r.change(self.artwork, inputs=inputs, outputs=outputs, queue=False, api_name=False)
            run_button.click(fn=self.artwork, inputs=inputs, outputs=outputs, api_name=False)
            image.change(self.artwork, inputs=inputs, outputs=outputs, queue=False, api_name=False)
        demo.queue()
        demo.launch(debug=True, share=True)


if __name__ == "__main__":
    run = Sketch2ImageLaunch()
    run.launch()
