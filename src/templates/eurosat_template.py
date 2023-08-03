eurosat_simple_template = [
    lambda c: f'a centered satellite photo of {c}.',
]

eurosat_main_template = [
    lambda c: f'a centered satellite photo of {c}',
]

eurosat_factor_templates = {
    "source": [
        "",
        ", by NASA",
        ', by Google Earth',
    ],

    "condition": [
        "",
        ", cool",
        ", nice",
        ", weird",
    ],
}

eurosat_factor_templates_gpt = {
    "source": [
        ", Yandex satellite",
        ", NASA",
        ", Google Maps",
        " ",
    ],
    "geographical_feature": [
        ", island",
        ", ul.",
        ", street",
        " ",
    ],
    "image_type": [
        ", satellite",
        ", aerial",
        ", map",
        " ",
    ],
    "natural_phenomenon": [
        ", hurricane",
        ", earthquake",
        ", deforestation",
        " ",
    ],
    "structure_type": [
        ", residential",
        ", commercial",
        ", fortress",
        " ",
    ]
}