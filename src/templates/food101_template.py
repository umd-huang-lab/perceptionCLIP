food_simple_template = [
    lambda c: f'a photo of {c}, a type of food.',
]

food_main_template = [
    lambda c: f'a photo of {c}, a type of food',
]


food_factor_templates = {
    "cuisines": {
        "others": [""],
        "african": ["African cuisine"],
        "american": ["American cuisine"],
        "asian": ["Asian cuisine"],
        "european": ["European cuisine"],
        "oceanic": ["Oceanic cuisine"],
    },
    "condition": {
        "normal": [""],
        "cool": ["cool", "stylish", "trendy"],
        "nice": ["nice", "pleasant", "good-looking"],
        "weird": ["weird", "odd", "unusual"],
    },
}