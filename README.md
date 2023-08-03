# More Context, Less Distraction: Visual Classification by Inferring and Conditioning on Contextual Attributes

by [Bang An*](https://bangann.github.io/), [Sicheng Zhu*](https://schzhu.github.io/)
, [Michael-Andrei Panaitescu-Liess](https://scholar.google.se/citations?user=MOP6lhkAAAAJ&hl=lv)
, [Chaithanya Kumar Mummadi](https://scholar.google.com/citations?user=XJLtaG4AAAAJ&hl=en)
, [Furong Huang](http://furong-huang.com/)

[[Paper](https://bangann.github.io/perceptionclip.pdf)] [[Twitter](https://twitter.com/furongh/status/1685424108748861440?s=20)]

## About

CLIP, as a foundational vision language model, is widely used in zero-shot image classification due
to its ability to understand various visual concepts and natural language descriptions. However, how
to fully leverage CLIP's unprecedented human-like understanding capabilities to achieve better
zero-shot classification is still an open question. This paper draws inspiration from the human
visual perception process: a modern neuroscience view suggests that in classifying an object, humans
first infer its class-independent attributes (e.g., background and orientation) which help separate
the foreground object from the background, and then make decisions based on this information.
Inspired by this, we observe that providing CLIP with contextual attributes improves zero-shot
classification and mitigates reliance on spurious features. We also observe that CLIP itself can
reasonably infer the attributes from an image. With these observations, we propose a training-free,
two-step zero-shot classification method named **PerceptionCLIP**. Given an image, it first infers
contextual attributes (e.g., background) and then performs object classification conditioning on
them. Our experiments show that PerceptionCLIP achieves better generalization, group robustness, and
better interpretability. For example, PerceptionCLIP with ViT-L/14 improves the worst group accuracy
by 16.5% on the Waterbirds dataset and by 3.5% on CelebA.

## Setup

### Setting up conda env (Optional)

```bash
conda create -n perceptionclip python=3.10
conda activate perceptionclip
pip install open_clip_torch
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch
pip install kornia
export PYTHONPATH="$PYTHONPATH:$PWD"
```

### Datasets

Please link all datasets to `./datasets/data`. Please refer to `DATA.md` for the structure of
dataset dictionaries.
Here's an example of how to create a symbolic link:

```bash
ln -s PATH_TO_YOUR_DATASET PATH_TO_YOUR_PROJ/datasets/data
```

### Code structure

Here's a brief intro of the major components of the code:

* `./src/datasets` contains the code for all the Datasets and Dataloaders.
* `./src/templates` contains all the text prompts.
* `./src/zero_shot_inference` contains the major code for our method and experiments.
* `./scripts` contains all the running scripts.
* `./replicate_runs.sh` This script calls other scripts in the `./scripts` directory with the
  necessary parameters to replicate our experiments.

To replicate all our experiments, please refer to `./replicate_runs.sh`. Below, we provide a few
examples demonstrating how to run the code using commands.

## CLIP understands contextual attributes

Followings are example commands to reproduce Figure 3.

```bash
# evaluate similarity score w/o z
python src/zero_shot_inference/eval_similarity.py --model=ViT-B/16 --eval_augmentation=vflip --template=simple_template  --save_name=sim_imagenet_vit16_wo

# evaluate similarity score w/ z_correct
python src/zero_shot_inference/eval_similarity.py --model=ViT-B/16 --eval_augmentation=vflip --template=vflip_template  --save_name=sim_imagenet_vit16_correct

# evaluate similarity score w/ z_wrong
python src/zero_shot_inference/eval_similarity.py --model=ViT-B/16 --eval_augmentation=vflip --template=vflip_template_wrong  --save_name=sim_imagenet_vit16_wrong

# evaluate similarity score w/ z_random
python src/zero_shot_inference/eval_similarity.py --model=ViT-B/16 --eval_augmentation=vflip --template=vflip_template  --random_descrip=True --save_name=sim_imagenet_vit16_random
```

## CLIP benefits from contextual attributes

Followings are example commands to reproduce Table 2.

```bash
# evaluate acc w/o z
python src/zero_shot_inference/zero_shot_org.py --model=ViT-B/16 --dataset=ImageNet --eval_augmentation=vflip --template=simple_template --save_name=acc_imagenet_vit16_wo

# evaluate acc w/ z_correct
python src/zero_shot_inference/zero_shot_org.py --model=ViT-B/16 --dataset=ImageNet --eval_augmentation=vflip --template=vflip_template --save_name=acc_imagenet_vit16_correct

# evaluate acc w/ z_wrong
python src/zero_shot_inference/zero_shot_org.py --model=ViT-B/16 --dataset=ImageNet --eval_augmentation=vflip --template=vflip_template_wrong --save_name=acc_imagenet_vit16_vflip_template_wrong

# evaluate acc w/ z_random
python src/zero_shot_inference/zero_shot_org.py --model=ViT-B/16 --dataset=ImageNet --eval_augmentation=vflip --template=vflip_template --random_descrip=True --save_name=acc_imagenet_vit16_random

# evaluate acc w/ self-inferred z
python src/zero_shot_inference/eval_acc_self_infer.py --model=ViT-B/16 --dataset=ImageNet --eval_augmentation=vflip --template0=vflip_template_wrong --template1=vflip_template --infer_mode=0 --save_name=acc_imagenet_vit16_self_infer_wy
```

## CLIP can infer contextual attributes

Followings are example commands to reproduce Table 3.

```bash
# method 1: w/ y
python src/zero_shot_inference/eval_infer_z.py --model=ViT-B/16 --dataset=ImageNet --eval_augmentation=vflip --template0=vflip_template_wrong --template1=vflip_template --infer_mode=0  --save_name=infer_z_imagenet_vit16_wy

# method 1: w/o y
python src/zero_shot_inference/eval_infer_z.py --model=ViT-B/16 --dataset=ImageNet --eval_augmentation=vflip --template0=vflip_template_wrong --template1=vflip_template --infer_mode=1  --save_name=infer_z_imagenet_vit16_woy
```

## PerceptionCLIP improves zero-shot generalization

Followings are example commands to reproduce Table 4.

```bash
# consider single attributes
python src/zero_shot_inference/perceptionclip_two_step.py --model=ViT-B/16 --dataset=ImageNet --main_template=imagenet_main_template --factor_templates=imagenet_factor_templates --factors=orientation --infer_mode=0 --temperature=3 --save_name=imagnet_ours_wy_vit16

# consider a composition of multiple attributes
python src/zero_shot_inference/perceptionclip_two_step.py --model=ViT-B/16 --dataset=ImageNet --main_template=imagenet_main_template --factor_templates=imagenet_factor_templates --factors=condition,quality --infer_mode=0 --temperature=3 --save_name=imagnet_ours_wy_vit16
```

Followings are example commands to reproduce Table 5.

```bash
# simple template
python src/zero_shot_inference/perceptionclip_one_step.py  --model=ViT-B/16 --dataset=CUB200  --template=simple_template --save_name=cub200_simple_vit16

# domain template
python src/zero_shot_inference/perceptionclip_one_step.py  --model=ViT-B/16 --dataset=CUB200  --template=cub200_simple_template --save_name=cub200_simple_vit16

# domain template + contextual attributes
python src/zero_shot_inference/perceptionclip_two_step.py --model=ViT-B/16 --dataset=CUB200 --main_template=cub200_main_template --factor_templates=cub200_factor_templates --factors=size,background,condition --convert_text=bird --infer_mode=0 --temperature=1 --save_name=cub200_ours_wy_vit16
```

## PerceptionCLIP improves group robustness

Followings are example commands to reproduce Table 7.

```bash
# w/ simple background
python src/zero_shot_inference/perceptionclip_two_step.py --model=RN50 --dataset=Waterbirds --template=waterbirds_background_template --infer_mode=0 --temperature=1 --eval_group=True --eval_trainset=True --save_name=waterbirds_ours_wy_RN50
  
# w/ complex background
python src/zero_shot_inference/perceptionclip_two_step.py --model=RN50 --dataset=Waterbirds --main_template=waterbirds_main_template --factor_templates=waterbirds_factor_templates --factors=background --infer_mode=0 --temperature=1 --eval_group=True --eval_trainset=True --save_name=waterbirds_ours_wy_RN50_factor
```

## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for releasing the source code.

* [CLIP](https://github.com/openai/CLIP)
* [OpenCLIP](https://github.com/mlfoundations/open_clip)
* [FLYP](https://github.com/locuslab/FLYP)

## Citing

If you find our work helpful, please cite it with:

```bibtex
@misc{an2023context,
      title={More Context, Less Distraction: Visual Classification by Inferring and Conditioning on Contextual Attributes}, 
      author={Bang An and Sicheng Zhu and Michael-Andrei Panaitescu-Liess and Chaithanya Kumar Mummadi and Furong Huang},
      year={2023},
      eprint={2308.01313},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```