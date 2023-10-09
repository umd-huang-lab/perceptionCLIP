flowers102_simple_template = [
    lambda c: f'a photo of a {c}, a type of flower.',
]

flowers102_main_template = [
    lambda c: f'a photo of a {c}, a type of flower'
]


flowers102_factor_templates = {
    "background": {
        "others": [""],
        "forest": ["in the forest"],
        "garden": ["in the garden"],
        "water": ["on water"],
        "dark_background": ["with dark background"],
    },
    "illumination": {
        "normal": [""],
        "bright": ["sunny", "bright"],
        "dark": ["dark", "dim"],
    },
    "condition": {
        "normal": [""],
        "cool": ["cool"],
        "nice": ["nice"],
        "weird": ["weird"],
    },
    "quality": {
        "others": [""],
        "high-res": ["high resolution"],
        "low-res": ["low resolution"],
    },
}