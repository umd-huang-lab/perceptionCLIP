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
    "simple_background": {
        "land": ["on land"],
        "water": ["on water"],
    },
    "background": {
        "others": [""],
        "land": ["on land"],
        "water": ["on water"],
        "forest": ["in forest"],
        "sky": ["in sky"],
        "street": ["on street"],
        "grass": ["on grass"],
        "tree": ["on tree"],
        "flowers": ["with flowers"],
        "beach": ["on beach"],
        "human": ["with human"],
        "branch": ["on a branch"],
    },

}
