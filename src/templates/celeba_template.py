celeba_simple_template = [
    lambda c: f"a photo of a celebrity with {c}."
]

celeba_main_template = [
    lambda c: f"a photo of a celebrity with {c}"
]

celeba_gender_template = [
    lambda c: f"a photo of a celebrity with {c}, female.",
    lambda c: f"a photo of a celebrity with {c}, male."
]


celeba_factor_templates = {
    "gender": {
        "not_sure": [""],
        "female": ["female"],
        "male": ["male"],
    },
    "age": {
        "not_sure": [""],
        "young": ["young"],
        "old": ["old"],
    },
    "race": {
        "others": [""],
        "white": ["white skin"],
        "dark": ["dark skin"],
        "asian": ["asian"],
    },
}