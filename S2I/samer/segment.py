import sys

sys.path.append("../../..")
sys.path.append("")
import cv2
import numpy as np
from S2I.samer.segmentor import Segmentor
from S2I.samer.transfer_tools import draw_outline, draw_points
from S2I.samer.seg_anything import draw_mask


class SegMent:
    def __init__(self, sam_args):
        self.sam = Segmentor(sam_args)
        self.reference_objs_list = []
        self.object_idx = 1
        self.curr_idx = 1
        self.origin_merged_mask = None  # init by segment-everything or update
        self.first_frame_mask = None

        # debug
        self.everything_points = []
        self.everything_labels = []
        print("SegTracker has been initialized")

    def seg_acc_bbox(self, origin_frame: np.ndarray, bbox: np.ndarray, ):
        # get interactive_mask
        interactive_mask = self.sam.segment_with_box(origin_frame, bbox)[0]
        refined_merged_mask = self.add_mask(interactive_mask)

        # draw mask
        masked_frame = draw_mask(origin_frame.copy(), refined_merged_mask)

        # draw bbox
        masked_frame = cv2.rectangle(masked_frame, bbox[0], bbox[1], (0, 0, 255))

        return refined_merged_mask, masked_frame

    def seg_acc_click(self, origin_frame: np.ndarray, coords: np.ndarray, modes: np.ndarray, multimask=True):
        # get interactive_mask
        interactive_mask = self.sam.segment_with_click(origin_frame, coords, modes, multimask)

        refined_merged_mask = self.add_mask(interactive_mask)

        # draw mask
        masked_frame = draw_mask(origin_frame.copy(), refined_merged_mask)
        masked_frame = draw_points(coords, modes, masked_frame)

        # draw outline
        masked_frame = draw_outline(interactive_mask, masked_frame)

        return refined_merged_mask, masked_frame

    def add_mask(self, interactive_mask: np.ndarray):
        if self.origin_merged_mask is None:
            self.origin_merged_mask = np.zeros(interactive_mask.shape, dtype=np.uint8)

        refined_merged_mask = self.origin_merged_mask.copy()
        refined_merged_mask[interactive_mask > 0] = self.curr_idx

        return refined_merged_mask

    def automatic_generate_mask(self, image):
        masks = self.sam.automatic_segment(image)
        return masks


if __name__ == '__main__':
    pass
