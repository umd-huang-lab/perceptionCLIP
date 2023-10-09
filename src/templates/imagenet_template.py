imagenet_main_template = [
    lambda c: f"a photo of a {c}"
]

imagenet_factor_templates = {
    "orientation": {
        "upright": ["", "upright"],
        "upside_down": ["upside-down", "flipped vertically", "inverted"],
        "rotated": ["rotated", "turned", "angled"],
    },
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
    "illumination": {
        "normal": [""],
        "bright": ["bright", "high exposure"],
        "dark": ["dark", "dim", "shadowy"],
    },
    "quantity": {
        "others": [""],
        "many": ["many", "several", "multiple"],
        "one": ["single", "one"],
        "large": ["large", "big", "huge"],
        "small": ["small", "tiny"],
    },
    "perspective": {
        "normal": [""],
        "close_up": ["close-up", "closeup"],
        "cropped": ["cropped"],
        "obscured": ["hard to see"],
    },
    "art": {
        "non_art": [""],
        "others": ["", "art", "drawing"],
        "sculpture": ["sculpture", "statue", "carving"],
        "rendering": ["digital rendering", "CGI", "digital art"],
        "graffiti": ["graffiti", "street art"],
        "tattoo": ["tattoo", "body art"],
        "embroidery": ["embroidery", "stitching", "needlework"],
        "paper_art": ["origami", "paper craft", "folded paper"],
        "sketch": ["sketch", "doodle", "scribble"],
        "cartoon": ["cartoon", "animation", "comic style"],
    },
    "medium": {
        "others": [""],
        "video_game": ["video game", "game scene", "digital"],
        "plastic": ["plastic", "synthetic"],
        "toy": ["toy", "plushie"],
    },
    "condition": {
        "normal": [""],
        "cool": ["cool", "stylish", "trendy"],
        "nice": ["nice", "pleasant", "good-looking"],
        "weird": ["weird", "odd", "unusual"],
    },
    "color_scheme": {
        "normal": [""],
        "bw": ["black and white", "monochrome", "grayscale"],
    },
    "tool": {
        "others": [""],
        "pencil": ["pencil sketch", "graphite drawing", "pencil drawn"],
        "pen": ["ink sketch", "pen drawing", "inked artwork"],
        "digital_tool": ["digital sketch", "computer generated", "digitally rendered"],
    },
}


