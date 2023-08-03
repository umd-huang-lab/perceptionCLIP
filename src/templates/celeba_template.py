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
    "gender": [
        "",
        ", female",
        ', male',
    ],
    "age": [
        "",
        ", young",
        ", old",
    ],
    "race": [
        "",
        ", white skin",
        ", dark skin",
        ", asian",
    ],
}
