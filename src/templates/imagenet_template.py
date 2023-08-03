imagenet_main_template = [
    lambda c: f"a photo of a {c}"
]

imagenet_factor_templates = {
    "orientation": [
        "",
        ", upside-down",
        ", rotated",
    ],
    "background": [
        "",
        ", in water",
        ", in forest",
        ", in sky",
        ", at street",
        ", at outdoor",
        ", at home",
        ", in office",
    ],
    "quality": [
        "",
        ", good",
        ", bad",
        ", low resolution",
        ", pixelated",
        ", jpeg corrupted",
        ", blurry",
        ", clean",
        ", dirty",
    ],
    "illumination": [
        "",
        ", bright",
        ", dark",
    ],
    "quantity": [
        "",
        ", many",
        ", one",
        ", large",
        ", small",
    ],
    "perspective": [
        "",
        ", close-up",
        ", cropped",
        ", hard to see",
    ],
    "art": [
        "",
        ", sculpture",
        ", rendering",
        ", graffiti",
        ", tattoo",
        ", embroidery",
        ", drawing",
        ", doodle",
        ", origami",
        ", sketch",
        ", art",
        ", cartoon",
    ],
    "medium": [
        "",
        ", video game",
        ", plastic",
        ", toy",
        ", plushie",
    ],
    "condition": [
        "",
        ", cool",
        ", nice",
        ", weird",
    ],
    "color-scheme": [
        "",
        ", black and white",
    ],
    "tool": [
        "",
        ", with pencil",
        ", with pen",
        ", digitally",
    ],

}

