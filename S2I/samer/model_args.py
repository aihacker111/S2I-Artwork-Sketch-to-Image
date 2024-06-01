def generate_sam_args(sam_checkpoint="ckpt", model_type="vit_b", points_per_side=16,
                      pred_iou_thresh=0.8, stability_score_thresh=0.9, crop_n_layers=1,
                      crop_n_points_downscale_factor=2, min_mask_region_area=200, gpu_id=0):
    sam_args = {
        'sam_checkpoint': f'{sam_checkpoint}/{model_type}.pth',
        'model_type': model_type,
        'generator_args': {
            'points_per_side': points_per_side,
            'pred_iou_thresh': pred_iou_thresh,
            'stability_score_thresh': stability_score_thresh,
            'crop_n_layers': crop_n_layers,
            'crop_n_points_downscale_factor': crop_n_points_downscale_factor,
            'min_mask_region_area': min_mask_region_area,
        },
        'gpu_id': gpu_id}

    return sam_args
