import gradio as gr
import random
import warnings
from S2I import Sketch2ImageController, css, scripts

warnings.filterwarnings("ignore")


class Sketch2ImageLaunch(Sketch2ImageController):
    def __init__(self):
        super().__init__(gr)

    def launch(self):
        with gr.Blocks(css=css) as demo:
            # gr.HTML(
            #     """
            #     <h1 style="display: flex; align-items: center; margin-bottom: 10px;">
            #     <img src="https://imgur.com/H2SLps2.png" alt="icon" style="margin-left: 10px; height: 30px;">
            #     S2I-Artwork <img src="https://imgur.com/cNMKSAy.png" alt="icon" style="margin-left: 10px; height: 30px;">: Personalized Sketch-to-Art 🧨 Diffusion Models
            #     <img src="https://imgur.com/yDnDd1p.png" alt="icon" style="margin-left: 10px; height: 30px;">
            #     </h1>
            #     <h3 style="margin-bottom: 10px;">Authors: Vo Nguyen An Tin, Nguyen Thiet Su</h3>
            #     <h4 style="margin-bottom: 10px;">This project is the fine-tuning task with LorA on large datasets included: COCO-2017, LHQ, Danbooru, LandScape and Mid-Journey V6</h4>
            #     <h4 style="margin-bottom: 10px;">We public 2 sketch2image-models-lora training on 30K and 60K steps with skip-connection and Transformers Super-Resolution variables</h4>
            #     <h4 style="margin-bottom: 10px;">View the full code project: <a href="https://github.com/aihacker111/S2I-Artwork-Sketch-to-Image/" target="_blank"> GitHub Repository</a></h4>
            #     <h4 style="margin-bottom: 10px;">
            #         <a href="https://github.com/aihacker111/S2I-Artwork-Sketch-to-Image/" target="_blank">
            #             <img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="100">
            #         </a>
            #     </h4>
            #     """
            # )
            gr.HTML(
                """
                <h1 style="display: flex; align-items: center; justify-content: center; margin-bottom: 10px; text-align: center;">
                <img src="https://imgur.com/H2SLps2.png" alt="icon" style="margin-left: 10px; height: 30px;">
                S2I-Artwork <img src="https://imgur.com/cNMKSAy.png" alt="icon" style="margin-left: 10px; height: 30px;">: Personalized Sketch-to-Art 🧨 Diffusion Models 
                <img src="https://imgur.com/yDnDd1p.png" alt="icon" style="margin-left: 10px; height: 30px;">
                </h1>
                <h3 style="text-align: center; margin-bottom: 10px;">Authors: Vo Nguyen An Tin, Nguyen Thiet Su</h3>
                <h4 style="margin-bottom: 10px;">*This project is the fine-tuning task with LorA on large datasets included: COCO-2017, LHQ, Danbooru, LandScape and Mid-Journey V6</h4>
                <h4 style="margin-bottom: 10px;">* We public 2 sketch2image-models-lora training on 30K and 60K steps with skip-connection and Transformers Super-Resolution variables</h4>
                <h4 style="margin-bottom: 10px;">* The inference and demo time of model is faster, you can slowly in the first runtime, but after that, the time process over 1.5 ~ 2s</h4>
                <h4 style="margin-bottom: 10px;">* View the full code project: <a href="https://github.com/aihacker111/S2I-Artwork-Sketch-to-Image/" target="_blank"> GitHub Repository</a></h4>
                <h4 style="margin-bottom: 10px;">
                    <a href="https://github.com/aihacker111/S2I-Artwork-Sketch-to-Image/" target="_blank">
                        <img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="100">
                    </a>
                </h4>
                """
            )
            with gr.Row(elem_id="main_row"):
                with gr.Column(elem_id="column_input"):
                    gr.Markdown("## SKETCH", elem_id="input_header")
                    image = gr.Sketchpad(
                        type="pil",
                        height=512,
                        width=512,
                        min_width=512,
                        image_mode="RGBA",
                        show_label=False,
                        mirror_webcam=False,
                        show_download_button=True,
                        elem_id='input_image',
                        brush=gr.Brush(colors=["#000000"], color_mode="fixed", default_size=4),
                        canvas_size=(1024, 1024),
                        layers=False
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

                with gr.Column(elem_id="column_output"):
                    gr.Markdown("## IMAGE GENERATE", elem_id="output_header")
                    result = gr.Image(
                        label="Result",
                        height=440,
                        width=440,
                        elem_id="output_image",
                        show_label=False,
                        show_download_button=True,
                    )
                    with gr.Row():
                        with gr.Column():
                            run_button = gr.Button("Generate 🪄", min_width=10, variant='primary')
                            prompt = gr.Textbox(label="Prompt", value="", show_label=True)
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
                        seed = gr.Textbox(label="Seed", value='42', scale=1, min_width=50)
                        randomize_seed = gr.Button(value='\U0001F3B2', variant='primary')
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
                        half_model = gr.Radio(
                            choices=["fp32", "fp16"],
                            value="fp32",
                            label="Demo Speed",
                            interactive=True)
                        model_options = gr.Radio(
                            choices=["30k", "60k"],
                            value="30k",
                            label="Type Sketch2Image models",
                            interactive=True)

                    # download_output = gr.Button("Download output", elem_id="download_output")
            demo.load(None, None, None, js=scripts)
            randomize_seed.click(
                lambda x: random.randint(0, self.MAX_SEED),
                inputs=[],
                outputs=seed,
                queue=False,
                api_name=False,
            )
            inputs = [image, prompt, prompt_temp, style, seed, val_r, half_model, model_options]
            outputs = [result, download_sketch]
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
