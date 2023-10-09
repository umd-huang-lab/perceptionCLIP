places_simple_template = [
    lambda c: f'a photo of a {c}.',
]

places_main_template = [
    lambda c: f'a photo of a {c}',
]


places_factor_templates = {
    "background": {
        "others": [""],
        "natural": ["in a natural setting", "outdoors in nature", "in the wild"],
        "urban": ["in an urban setting", "city background"],
        "indoor": ["indoors", "inside a building", "enclosed space"],
    },
    "quality": {
        "normal": [""],
        "good": ["good", "looks good"],
        "bad": ["bad", "poor"],
        "low_res": ["low resolution"],
        "pixelated": ["pixelated"],
        "jpeg_corrupted": ["jpeg corrupted", "jpeg"],
        "blurry": ["blurry"],
        "clean": ["clean"],
        "dirty": ["dirty"],
    },
    "condition": {
        "normal": [""],
        "cool": ["cool", "stylish", "trendy"],
        "nice": ["nice", "pleasant", "good-looking"],
        "weird": ["weird", "odd", "unusual"],
    },
}
