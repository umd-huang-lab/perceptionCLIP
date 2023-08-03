waterbirds_simple_template = [
    lambda c: f"a photo of a {c}."
]

waterbirds_main_template = [
    lambda c: f"a photo of a {c}"
]

waterbirds_background_template = [
    lambda c: f"a photo of a {c}, on land.",
    lambda c: f"a photo of a {c}, on water.",
]

waterbirds_factor_templates = {
    "background": [
        "",
        ", on land",
        ", on water",
        ", in forest",
        ", in sky",
        ", on street",
        ", on grass",
        ", on tree",
        ", with flowers",
        ", on beach",
        ", with human",
        ", on a branch",
    ],
}
