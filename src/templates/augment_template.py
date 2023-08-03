
vflip_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, upside-down.",
    lambda c: f"a photo of a {c}, the photo is upside-down.",
]

vflip_template_wrong = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, upright.",
    lambda c: f"a photo of a {c}, the photo is upright.",
]

rotation_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, rotated.",
    lambda c: f"a photo of a {c}, the photo is rotated.",
]

rotation_template_wrong = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, upright.",
    lambda c: f"a photo of a {c}, the photo is upright.",
]


elastic_transform_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, distorted.",
    lambda c: f"a photo of a {c}, the photo is distorted.",
]

elastic_transform_template_wrong = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, normal.",
    lambda c: f"a photo of a {c}, the photo is normal.",
]

invert_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, color-inverted.",
    lambda c: f"a photo of a {c}, the photo is color-inverted.",
]

invert_template_wrong = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, normal.",
    lambda c: f"a photo of a {c}, the photo is normal.",
]

solarize_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, solarized.",
    lambda c: f"a photo of a {c}, the photo is solarized.",
]

solarize_template_wrong = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, normal.",
    lambda c: f"a photo of a {c}, the photo is normal.",
]

blur_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, blurred.",
    lambda c: f"a photo of a {c}, the photo is blurred.",
]

blur_template_wrong = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, clear.",
    lambda c: f"a photo of a {c}, the photo is clear.",
]


grayscale_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, grayscale.",
    lambda c: f"a photo of a {c}, the photo is in black and white.",
]

grayscale_template_wrong = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, colorful.",
    lambda c: f"a photo of a {c}, the photo is colorful.",
]


bright_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, bright.",
    lambda c: f"a photo of a {c}, the photo is bright.",
]


bright_template_wrong = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, dark.",
    lambda c: f"a photo of a {c}, the photo is dark.",
]

noise_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, with noise.",
    lambda c: f"a photo of a {c}, the photo has noise.",
]

noise_template_wrong = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, clear.",
    lambda c: f"a photo of a {c}, the photo is clear.",
]

snow_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, in the snow.",
    lambda c: f"a photo of a {c}, the photo is in the snow.",
]

snow_template_wrong = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, clear.",
    lambda c: f"a photo of a {c}, the photo is clear.",
]

frost_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, in the frost.",
    lambda c: f"a photo of a {c}, the photo is in the frost.",
]

frost_template_wrong = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, clear.",
    lambda c: f"a photo of a {c}, the photo is clear.",
]

fog_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, in the fog.",
    lambda c: f"a photo of a {c}, the photo is in the fog.",
]

fog_template_wrong = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, clear.",
    lambda c: f"a photo of a {c}, the photo is clear.",
]

jpeg_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, in jpeg format.",
    lambda c: f"a photo of a {c}, the photo is in jpeg format.",
]

jpeg_template_wrong = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, in high resolution.",
    lambda c: f"a photo of a {c}, the photo is in high resolution.",
]

vflip_invert_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, upside-down, color-inverted.",
    lambda c: f"a photo of a {c}, the photo is upside-down, color-inverted.",
]

vflip_invert_template_wrong = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, upright, normal.",
    lambda c: f"a photo of a {c}, the photo is upright, normal.",
]


grayscale_elastic_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, grayscale, with distortion.",
    lambda c: f"a photo of a {c}, the photo is distorted, in black and white.",
]

grayscale_elastic_template_wrong = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of a {c}, colorful, normal.",
    lambda c: f"a photo of a {c}, the photo is colorful, normal.",
]

