oxfordpets_simple_template = [
    lambda c: f'a photo of a {c}, a type of pet.',
]

oxfordpets_main_template = [
    lambda c: f'a photo of a {c}, a type of pet',
]

oxfordpets_factor_templates = {
    "species": [
        "",
        ", dog",
        ", cat",
    ],
    "background": [
        "",
        ", indoors",
        ", outdoors",
        ", on a bed",
        ", on a couch",
        ", at the beach",
        ", in a park",
        ", on grass",
        ", in a tree",
    ],
    "pose": [
        "",
        ", sitting",
        ", running",
        ", sleeping",
        ", eating",
        ", playing",
    ],
    "interaction": [
        "",
        ", interacting with another pet",
        ", interacting with a person",
        ", playing with a toy",
        ", being held",
        ", being petted",
    ],
}
