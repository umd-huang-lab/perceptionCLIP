flowers102_simple_template = [
    lambda c: f'a photo of a {c}, a type of flower.',
]

flowers102_main_template = [
    lambda c: f'a photo of a {c}, a type of flower'
]

flowers102_factor_templates = {
    "background": [
        "",
        ", in the forest",
        ', in the garden',
        ", on water",
        ", with dark background",
    ],
    "illumination": [
        "",
        ", sunny",
        ", dark",
    ],
    "condition": [
        "",
        ", cool",
        ", nice",
        ", weird",
    ],
    "quality": [
        "",
        ", low resolution",
        ", high resolution",
    ],
}