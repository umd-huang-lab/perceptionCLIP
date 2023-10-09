oxfordpets_simple_template = [
    lambda c: f'a photo of a {c}, a type of pet.',
]

oxfordpets_main_template = [
    lambda c: f'a photo of a {c}, a type of pet',
]

oxfordpets_factor_templates = {
    "species": {
        "others": [""],
        "dog": ["dog"],
        "cat": ["cat"],
    },
    "background": {
        "others": [""],
        "indoors": ["indoors"],
        "outdoors": ["outdoors"],
        "bed": ["on a bed"],
        "couch": ["on a couch"],
        "beach": ["at the beach"],
        "park": ["in a park"],
        "grass": ["on grass"],
        "tree": ["on a tree"],
    },
    "pose": {
        "others": [""],
        "sitting": ["sitting"],
        "running": ["running"],
        "sleeping": ["sleeping"],
        "eating": ["eating"],
        "playing": ["playing"],
    },
    "interaction": {
        "others": [""],
        "pet_interaction": ["interacting with another pet"],
        "human_interaction": ["interacting with a person"],
        "toy": ["playing with a toy"],
        "held": ["being held"],
        "petted": ["being petted"],
    },
}
