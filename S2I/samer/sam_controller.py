from S2I.samer import SegMent, generate_sam_args
from S2I.logger import logger
from tqdm import tqdm
import gradio as gr
import numpy as np
import os
import shutil
import cv2
import requests


class SAMController:
    def __init__(self):
        self.current_model_type = None
        self.refine_mask = None

    @staticmethod
    def clean():
        return None, None, None, None, None, [[]]

    @staticmethod
    def save_mask(refined_mask=None, save=False):

        if refined_mask is not None and save:
            if os.path.exists(os.path.join(os.getcwd(), 'output_render')):
                shutil.rmtree(os.path.join(os.getcwd(), 'output_render'))
            save_path = os.path.join(os.getcwd(), 'output_render')
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, f'refined_mask_result.png'), (refined_mask * 255).astype('uint8'))
        elif refined_mask is None and save:
            return os.path.join(os.path.join(os.getcwd(), 'output_render'), f'refined_mask_result.png')

    @staticmethod
    def download_models(model_type):
        dir_path = os.path.join(os.getcwd(), 'root_model')
        sam_models_path = os.path.join(dir_path, 'sam_models')

        # Models URLs
        models_urls = {
            'sam_models': {
                'vit_b': 'https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_b_01ec64.pth?download=true',
                'vit_l': 'https://huggingface.co/segments-arnaud/sam_vit_l/resolve/main/sam_vit_l_0b3195.pth?download=true',
                'vit_h': 'https://huggingface.co/segments-arnaud/sam_vit_h/resolve/main/sam_vit_h_4b8939.pth?download=true'
            }
        }

        # Download specified model type
        if model_type in models_urls['sam_models']:
            model_url = models_urls['sam_models'][model_type]
            os.makedirs(sam_models_path, exist_ok=True)
            model_path = os.path.join(sam_models_path, model_type + '.pth')

            if not os.path.exists(model_path):
                logger.info(f"Downloading {model_type} model...")
                response = requests.get(model_url, stream=True)
                response.raise_for_status()  # Raise an exception for non-2xx status codes

                total_size = int(response.headers.get('content-length', 0))  # Get file size from headers
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading {model_type} model") as pbar:
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            f.write(chunk)
                            pbar.update(len(chunk))
                logger.info(f"{model_type} model downloaded.")
            else:
                logger.info(f"{model_type} model already exists.")
            return logger.info(f"{model_type} model download complete.")
        else:
            return logger.info(f"Invalid model type: {model_type}")

    @staticmethod
    def get_models_path(model_type=None, segment=False):
        sam_models_path = os.path.join(os.getcwd(), 'root_model', 'sam_models')

        if segment:
            sam_args = generate_sam_args(sam_checkpoint=sam_models_path, model_type=model_type)
            return sam_args, sam_models_path

    @staticmethod
    def get_click_prompt(click_stack, point):
        click_stack[0].append(point["coord"])
        click_stack[1].append(point["mode"]
                              )

        prompt = {
            "points_coord": click_stack[0],
            "points_mode": click_stack[1],
            "multi_mask": "True",
        }

        return prompt

    @staticmethod
    def read_temp_file(temp_file_wrapper):
        name = temp_file_wrapper.name
        with open(temp_file_wrapper.name, 'rb') as f:
            # Read the content of the file
            file_content = f.read()
        return file_content, name

    def get_meta_from_image(self, input_img):
        file_content, _ = self.read_temp_file(input_img)
        np_arr = np.frombuffer(file_content, np.uint8)

        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        first_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return first_frame, first_frame

    def is_sam_model(self, model_type):
        sam_args, sam_models_dir = self.get_models_path(model_type=model_type, segment=True)
        model_path = os.path.join(sam_models_dir, model_type + '.pth')
        if not os.path.exists(model_path):
            self.download_models(model_type=model_type)
            return 'Model is downloaded', sam_args
        else:
            return 'Model is already downloaded', sam_args

    @staticmethod
    def init_segment(
            points_per_side,
            origin_frame,
            sam_args,
            predict_iou_thresh=0.8,
            stability_score_thresh=0.9,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=200):
        if origin_frame is None:
            return None, origin_frame, [[], []]
        sam_args["generator_args"]["points_per_side"] = points_per_side
        sam_args["generator_args"]["pred_iou_thresh"] = predict_iou_thresh
        sam_args["generator_args"]["stability_score_thresh"] = stability_score_thresh
        sam_args["generator_args"]["crop_n_layers"] = crop_n_layers
        sam_args["generator_args"]["crop_n_points_downscale_factor"] = crop_n_points_downscale_factor
        sam_args["generator_args"]["min_mask_region_area"] = min_mask_region_area

        segment = SegMent(sam_args)
        logger.info(f"Model Init: {sam_args}")
        return segment, origin_frame, [[], []]

    @staticmethod
    def seg_acc_click(segment, prompt, origin_frame):
        # seg acc to click
        refined_mask, masked_frame = segment.seg_acc_click(
            origin_frame=origin_frame,
            coords=np.array(prompt["points_coord"]),
            modes=np.array(prompt["points_mode"]),
            multimask=prompt["multi_mask"],
        )
        return refined_mask, masked_frame

    def undo_click_stack_and_refine_seg(self, segment, origin_frame, click_stack):
        if segment is None:
            return segment, origin_frame, [[], []]

        logger.info("Undo !")
        if len(click_stack[0]) > 0:
            click_stack[0] = click_stack[0][: -1]
            click_stack[1] = click_stack[1][: -1]

        if len(click_stack[0]) > 0:
            prompt = {
                "points_coord": click_stack[0],
                "points_mode": click_stack[1],
                "multi_mask": "True",
            }

            _, masked_frame = self.seg_acc_click(segment, prompt, origin_frame)
            return segment, masked_frame, click_stack
        else:
            return segment, origin_frame, [[], []]

    def reload_segment(self,
                       check_sam,
                       segment,
                       model_type,
                       point_per_sides,
                       origin_frame,
                       predict_iou_thresh,
                       stability_score_thresh,
                       crop_n_layers,
                       crop_n_points_downscale_factor,
                       min_mask_region_area):
        status, sam_args = check_sam(model_type)
        if segment is None or status == 'Model is downloaded':
            segment, _, _ = self.init_segment(point_per_sides,
                                              origin_frame,
                                              sam_args,
                                              predict_iou_thresh,
                                              stability_score_thresh,
                                              crop_n_layers,
                                              crop_n_points_downscale_factor,
                                              min_mask_region_area)
            self.current_model_type = model_type
        return segment, self.current_model_type, status

    def sam_click(self,
                  evt: gr.SelectData,
                  segment,
                  origin_frame,
                  model_type,
                  point_mode,
                  click_stack,
                  point_per_sides,
                  predict_iou_thresh,
                  stability_score_thresh,
                  crop_n_layers,
                  crop_n_points_downscale_factor,
                  min_mask_region_area):
        logger.info("Click")
        if point_mode == "Positive":
            point = {"coord": [evt.index[0], evt.index[1]], "mode": 1}
        else:
            point = {"coord": [evt.index[0], evt.index[1]], "mode": 0}
        click_prompt = self.get_click_prompt(click_stack, point)
        segment, self.current_model_type, status = self.reload_segment(
            self.is_sam_model,
            segment,
            model_type,
            point_per_sides,
            origin_frame,
            predict_iou_thresh,
            stability_score_thresh,
            crop_n_layers,
            crop_n_points_downscale_factor,
            min_mask_region_area)
        if segment is not None and model_type != self.current_model_type:
            segment = None
            segment, _, status = self.reload_segment(
                self.is_sam_model,
                segment,
                model_type,
                point_per_sides,
                origin_frame,
                predict_iou_thresh,
                stability_score_thresh,
                crop_n_layers,
                crop_n_points_downscale_factor,
                min_mask_region_area)
        refined_mask, masked_frame = self.seg_acc_click(segment, click_prompt, origin_frame)
        self.save_mask(refined_mask, save=True)
        self.refine_mask = refined_mask
        return segment, masked_frame, click_stack, status

    @staticmethod
    def normalize_image(image):
        # Normalize the image to the range [0, 1]
        min_val = image.min()
        max_val = image.max()
        image = (image - min_val) / (max_val - min_val)

        return image

    @staticmethod
    def compute_probability(masks):
        p_max = None
        for mask in masks:
            p = mask['prob']
            if p_max is None:
                p_max = p
            else:
                p_max = np.maximum(p_max, p)
        return p_max
    @staticmethod
    def download_opencv_model(model_url):
        opencv_model_path = os.path.join(os.getcwd(), 'edges_detection')
        os.makedirs(opencv_model_path, exist_ok=True)
        model_path = os.path.join(opencv_model_path, 'edges_detection' + '.yml.gz')
        response = requests.get(model_url, stream=True)
        response.raise_for_status()  # Raise an exception for non-2xx status codes

        total_size = int(response.headers.get('content-length', 0))  # Get file size from headers
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading opencv model") as pbar:
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return model_path

    def automatic_sam2sketch(self,
                             segment,
                             image,
                             origin_frame,
                             model_type
                             ):
        _, sam_args = self.is_sam_model(model_type)
        if segment is None or model_type != sam_args['model_type']:
            segment, _, _ = self.init_segment(
                points_per_side=16,
                origin_frame=origin_frame,
                sam_args=sam_args,
                predict_iou_thresh=0.8,
                stability_score_thresh=0.9,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=200)
        model_path = self.download_opencv_model(model_url='https://github.com/nipunmanral/Object-Detection-using-OpenCV/raw/master/model.yml.gz')
        masks = segment.automatic_generate_mask(image)
        p_max = self.compute_probability(masks)
        edges = self.normalize_image(p_max)
        edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model_path)
        orimap = edge_detection.computeOrientation(edges)
        edges = edge_detection.edgesNms(edges, orimap)
        edges = (edges * 255).astype('uint8')
        edges = 255 - edges
        edges = np.stack((edges,) * 3, axis=-1)
        return edges
