cub200_simple_template = [
    lambda c: f'a photo of a {c}, a type of bird.',
]

cub200_main_template = [
    lambda c: f'a photo of a {c}, a type of bird'
]


cub200_factor_templates = {
    "size": {
        "others": [""],
        "small": ["small"],
        "big": ["big"],
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
    "condition": {
        "normal": [""],
        "cool": ["cool"],
        "nice": ["nice"],
        "weird": ["weird"],
    },

}

