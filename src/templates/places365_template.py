places_simple_template = [
    lambda c: f'a photo of a {c}.',
]

places_main_template = [
    lambda c: f'a photo of a {c}',
]

places_factor_templates = {
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
    "condition": [
        "",
        ", cool",
        ", nice",
        ", weird",
    ],
}
