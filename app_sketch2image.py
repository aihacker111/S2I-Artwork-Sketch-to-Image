import gradio as gr
import random
from S2I import Sketch2ImageController, css, scripts


class Sketch2ImageLaunch(Sketch2ImageController):
    def __init__(self):
        super().__init__(gr)

    def launch(self):
        with gr.Blocks(css=css) as demo:
            line = gr.Checkbox(label="line", value=False, elem_id="cb-line")
            eraser = gr.Checkbox(label="eraser", value=False, elem_id="cb-eraser")
            with gr.Row(elem_id="main_row"):
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
                    with gr.Row():
                        val_r = gr.Slider(
                            label="Sketch guidance: ",
                            show_label=True,
                            minimum=0,
                            maximum=1,
                            value=0.4,
                            step=0.01,
                            scale=3,
                        )
                        seed = gr.Textbox(label="Seed", value=42, scale=1, min_width=50)
                        randomize_seed = gr.Button(value='\U0001F3B2')

                with gr.Column(elem_id="column_process", min_width=50, scale=0.4):
                    run_button = gr.Button("Run", min_width=50)

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
                download_output = gr.Button("Download output", elem_id="download_output")
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
            inputs = [image, prompt, prompt_temp, style, seed, val_r]
            outputs = [result, download_sketch, download_output]
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
