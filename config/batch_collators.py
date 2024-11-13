from batch_collators import norm_img, img_channels_first


def default_ijepa_multiblock_collator(
    img_size=224,
    patch_size=16,
):
    return {
        "input_size": (img_size, img_size),
        "patch_size": patch_size,
        "enc_mask_scale": (0.85, 1.0),
        "pred_mask_scale": (0.15, 0.2),
        "aspect_ratio": (0.75, 1.5),
        "nenc": 1,
        "npred": 4,
        "min_keep": 10,
        "allow_overlap": False,
        "data_transforms": [norm_img, img_channels_first],
    }


def default_classification_collator():
    return {"data_transforms": [norm_img, img_channels_first]}
