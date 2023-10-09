###################### CLIP understands contextual attributes (Fig 3) ######################
# w/o z
sh scripts/srun_eval_sim.sh vflip             simple_template  sim_imagenet_vit16_wo
sh scripts/srun_eval_sim.sh rotation          simple_template  sim_imagenet_vit16_wo
sh scripts/srun_eval_sim.sh elastic_transform simple_template  sim_imagenet_vit16_wo
sh scripts/srun_eval_sim.sh invert            simple_template  sim_imagenet_vit16_wo
sh scripts/srun_eval_sim.sh solarize          simple_template  sim_imagenet_vit16_wo
sh scripts/srun_eval_sim.sh blur              simple_template  sim_imagenet_vit16_wo
sh scripts/srun_eval_sim.sh grayscale         simple_template  sim_imagenet_vit16_wo
sh scripts/srun_eval_sim.sh bright            simple_template  sim_imagenet_vit16_wo

sh scripts/srun_eval_sim_c.sh noise  simple_template sim_imagenet_vit16_wo ./datasets/data/imagenet-c/shot_noise/5
sh scripts/srun_eval_sim_c.sh snow   simple_template sim_imagenet_vit16_wo ./datasets/data/imagenet-c/snow/5
sh scripts/srun_eval_sim_c.sh frost  simple_template sim_imagenet_vit16_wo ./datasets/data/imagenet-c/frost/5
sh scripts/srun_eval_sim_c.sh fog    simple_template sim_imagenet_vit16_wo ./datasets/data/imagenet-c/fog/5
sh scripts/srun_eval_sim_c.sh jpeg   simple_template sim_imagenet_vit16_wo ./datasets/data/imagenet-c/jpeg_compression/5

# w/ z_correct
sh scripts/srun_eval_sim.sh vflip             vflip_template             sim_imagenet_vit16_correct
sh scripts/srun_eval_sim.sh rotation          rotation_template          sim_imagenet_vit16_correct
sh scripts/srun_eval_sim.sh elastic_transform elastic_transform_template sim_imagenet_vit16_correct
sh scripts/srun_eval_sim.sh invert            invert_template            sim_imagenet_vit16_correct
sh scripts/srun_eval_sim.sh solarize          solarize_template          sim_imagenet_vit16_correct
sh scripts/srun_eval_sim.sh blur              blur_template              sim_imagenet_vit16_correct
sh scripts/srun_eval_sim.sh grayscale         grayscale_template         sim_imagenet_vit16_correct
sh scripts/srun_eval_sim.sh bright            bright_template            sim_imagenet_vit16_correct

sh scripts/srun_eval_sim_c.sh noise  noise_template sim_imagenet_vit16_correct ./datasets/data/imagenet-c/shot_noise/5
sh scripts/srun_eval_sim_c.sh snow   snow_template  sim_imagenet_vit16_correct ./datasets/data/imagenet-c/snow/5
sh scripts/srun_eval_sim_c.sh frost  frost_template sim_imagenet_vit16_correct ./datasets/data/imagenet-c/frost/5
sh scripts/srun_eval_sim_c.sh fog    fog_template   sim_imagenet_vit16_correct ./datasets/data/imagenet-c/fog/5
sh scripts/srun_eval_sim_c.sh jpeg   jpeg_template  sim_imagenet_vit16_correct ./datasets/data/imagenet-c/jpeg_compression/5

# w/ z_wrong
sh scripts/srun_eval_sim.sh vflip             vflip_template_wrong             sim_imagenet_vit16_wrong
sh scripts/srun_eval_sim.sh rotation          rotation_template_wrong          sim_imagenet_vit16_wrong
sh scripts/srun_eval_sim.sh elastic_transform elastic_transform_template_wrong sim_imagenet_vit16_wrong
sh scripts/srun_eval_sim.sh invert            invert_template_wrong            sim_imagenet_vit16_wrong
sh scripts/srun_eval_sim.sh solarize          solarize_template_wrong          sim_imagenet_vit16_wrong
sh scripts/srun_eval_sim.sh blur              blur_template_wrong              sim_imagenet_vit16_wrong
sh scripts/srun_eval_sim.sh grayscale         grayscale_template_wrong         sim_imagenet_vit16_wrong
sh scripts/srun_eval_sim.sh bright            bright_template_wrong            sim_imagenet_vit16_wrong

sh scripts/srun_eval_sim_c.sh noise  noise_template_wrong sim_imagenet_vit16_wrong ./datasets/data/imagenet-c/shot_noise/5
sh scripts/srun_eval_sim_c.sh snow   snow_template_wrong  sim_imagenet_vit16_wrong ./datasets/data/imagenet-c/snow/5
sh scripts/srun_eval_sim_c.sh frost  frost_template_wrong sim_imagenet_vit16_wrong ./datasets/data/imagenet-c/frost/5
sh scripts/srun_eval_sim_c.sh fog    fog_template_wrong   sim_imagenet_vit16_wrong ./datasets/data/imagenet-c/fog/5
sh scripts/srun_eval_sim_c.sh jpeg   jpeg_template_wrong  sim_imagenet_vit16_wrong ./datasets/data/imagenet-c/jpeg_compression/5

# w/ z_random
sh scripts/srun_eval_sim_random.sh vflip             vflip_template             sim_imagenet_vit16_random
sh scripts/srun_eval_sim_random.sh rotation          rotation_template          sim_imagenet_vit16_random
sh scripts/srun_eval_sim_random.sh elastic_transform elastic_transform_template sim_imagenet_vit16_random
sh scripts/srun_eval_sim_random.sh invert            invert_template            sim_imagenet_vit16_random
sh scripts/srun_eval_sim_random.sh solarize          solarize_template          sim_imagenet_vit16_random
sh scripts/srun_eval_sim_random.sh blur              blur_template              sim_imagenet_vit16_random
sh scripts/srun_eval_sim_random.sh grayscale         grayscale_template         sim_imagenet_vit16_random
sh scripts/srun_eval_sim_random.sh bright            bright_template            sim_imagenet_vit16_random

sh scripts/srun_eval_sim_random_c.sh noise  noise_template sim_imagenet_vit16_random ./datasets/data/imagenet-c/shot_noise/5
sh scripts/srun_eval_sim_random_c.sh snow   snow_template  sim_imagenet_vit16_random ./datasets/data/imagenet-c/snow/5
sh scripts/srun_eval_sim_random_c.sh frost  frost_template sim_imagenet_vit16_random ./datasets/data/imagenet-c/frost/5
sh scripts/srun_eval_sim_random_c.sh fog    fog_template   sim_imagenet_vit16_random ./datasets/data/imagenet-c/fog/5
sh scripts/srun_eval_sim_random_c.sh jpeg   jpeg_template  sim_imagenet_vit16_random ./datasets/data/imagenet-c/jpeg_compression/5


###################### CLIP benefits from contextual attributes (Table 2) ######################
# w/o z
sh scripts/srun_eval_acc.sh vflip             simple_template     ImageNet        acc_imagenet_vit16_wo
sh scripts/srun_eval_acc.sh rotation          simple_template     ImageNet        acc_imagenet_vit16_wo
sh scripts/srun_eval_acc.sh elastic_transform simple_template     ImageNet        acc_imagenet_vit16_wo
sh scripts/srun_eval_acc.sh invert            simple_template     ImageNet        acc_imagenet_vit16_wo
sh scripts/srun_eval_acc.sh solarize          simple_template     ImageNet        acc_imagenet_vit16_wo
sh scripts/srun_eval_acc.sh blur              simple_template     ImageNet        acc_imagenet_vit16_wo
sh scripts/srun_eval_acc.sh grayscale         simple_template     ImageNet        acc_imagenet_vit16_wo
sh scripts/srun_eval_acc.sh bright            simple_template     ImageNet        acc_imagenet_vit16_wo

sh scripts/srun_eval_acc_c.sh noise  simple_template  acc_imagenet_vit16_wo ./datasets/data/imagenet-c/shot_noise/5
sh scripts/srun_eval_acc_c.sh snow   simple_template  acc_imagenet_vit16_wo ./datasets/data/imagenet-c/snow/5
sh scripts/srun_eval_acc_c.sh frost  simple_template  acc_imagenet_vit16_wo ./datasets/data/imagenet-c/frost/5
sh scripts/srun_eval_acc_c.sh fog    simple_template  acc_imagenet_vit16_wo ./datasets/data/imagenet-c/fog/5
sh scripts/srun_eval_acc_c.sh jpeg   simple_template  acc_imagenet_vit16_wo ./datasets/data/imagenet-c/jpeg_compression/5

# w/ z_correct
sh scripts/srun_eval_acc.sh vflip             vflip_template              ImageNet   acc_imagenet_vit16_correct
sh scripts/srun_eval_acc.sh rotation          rotation_template           ImageNet   acc_imagenet_vit16_correct
sh scripts/srun_eval_acc.sh elastic_transform elastic_transform_template  ImageNet   acc_imagenet_vit16_correct
sh scripts/srun_eval_acc.sh invert            invert_template             ImageNet   acc_imagenet_vit16_correct
sh scripts/srun_eval_acc.sh solarize          solarize_template           ImageNet   acc_imagenet_vit16_correct
sh scripts/srun_eval_acc.sh blur              blur_template               ImageNet   acc_imagenet_vit16_correct
sh scripts/srun_eval_acc.sh grayscale         grayscale_template          ImageNet   acc_imagenet_vit16_correct
sh scripts/srun_eval_acc.sh bright            bright_template             ImageNet   acc_imagenet_vit16_correct

sh scripts/srun_eval_acc_c.sh noise  noise_template acc_imagenet_vit16_correct ./datasets/data/imagenet-c/shot_noise/5
sh scripts/srun_eval_acc_c.sh snow   snow_template  acc_imagenet_vit16_correct ./datasets/data/imagenet-c/snow/5
sh scripts/srun_eval_acc_c.sh frost  frost_template acc_imagenet_vit16_correct ./datasets/data/imagenet-c/frost/5
sh scripts/srun_eval_acc_c.sh fog    fog_template   acc_imagenet_vit16_correct ./datasets/data/imagenet-c/fog/5
sh scripts/srun_eval_acc_c.sh jpeg   jpeg_template  acc_imagenet_vit16_correct ./datasets/data/imagenet-c/jpeg_compression/5

# w/ z_wrong
sh scripts/srun_eval_acc.sh vflip             vflip_template_wrong                ImageNet  acc_imagenet_vit16_wrong
sh scripts/srun_eval_acc.sh rotation          rotation_template_wrong             ImageNet  acc_imagenet_vit16_wrong
sh scripts/srun_eval_acc.sh elastic_transform elastic_transform_template_wrong    ImageNet  acc_imagenet_vit16_wrong
sh scripts/srun_eval_acc.sh invert            invert_template_wrong               ImageNet  acc_imagenet_vit16_wrong
sh scripts/srun_eval_acc.sh solarize          solarize_template_wrong             ImageNet  acc_imagenet_vit16_wrong
sh scripts/srun_eval_acc.sh blur              blur_template_wrong                 ImageNet  acc_imagenet_vit16_wrong
sh scripts/srun_eval_acc.sh grayscale         grayscale_template_wrong            ImageNet  acc_imagenet_vit16_wrong
sh scripts/srun_eval_acc.sh bright            bright_template_wrong               ImageNet  acc_imagenet_vit16_wrong

sh scripts/srun_eval_acc_c.sh noise  noise_template_wrong acc_imagenet_vit16_wrong ./datasets/data/imagenet-c/shot_noise/5
sh scripts/srun_eval_acc_c.sh snow   snow_template_wrong  acc_imagenet_vit16_wrong ./datasets/data/imagenet-c/snow/5
sh scripts/srun_eval_acc_c.sh frost  frost_template_wrong acc_imagenet_vit16_wrong ./datasets/data/imagenet-c/frost/5
sh scripts/srun_eval_acc_c.sh fog    fog_template_wrong   acc_imagenet_vit16_wrong ./datasets/data/imagenet-c/fog/5
sh scripts/srun_eval_acc_c.sh jpeg   jpeg_template_wrong  acc_imagenet_vit16_wrong ./datasets/data/imagenet-c/jpeg_compression/5

# w/ z_random
sh scripts/srun_eval_acc_random.sh vflip             vflip_template             acc_imagenet_vit16_random
sh scripts/srun_eval_acc_random.sh rotation          rotation_template          acc_imagenet_vit16_random
sh scripts/srun_eval_acc_random.sh elastic_transform elastic_transform_template acc_imagenet_vit16_random
sh scripts/srun_eval_acc_random.sh invert            invert_template            acc_imagenet_vit16_random
sh scripts/srun_eval_acc_random.sh solarize          solarize_template          acc_imagenet_vit16_random
sh scripts/srun_eval_acc_random.sh blur              blur_template              acc_imagenet_vit16_random
sh scripts/srun_eval_acc_random.sh grayscale         grayscale_template         acc_imagenet_vit16_random
sh scripts/srun_eval_acc_random.sh bright            bright_template            acc_imagenet_vit16_random

sh scripts/srun_eval_acc_random_c.sh noise  noise_template acc_imagenet_vit16_random ./datasets/data/imagenet-c/shot_noise/5
sh scripts/srun_eval_acc_random_c.sh snow   snow_template  acc_imagenet_vit16_random ./datasets/data/imagenet-c/snow/5
sh scripts/srun_eval_acc_random_c.sh frost  frost_template acc_imagenet_vit16_random ./datasets/data/imagenet-c/frost/5
sh scripts/srun_eval_acc_random_c.sh fog    fog_template   acc_imagenet_vit16_random ./datasets/data/imagenet-c/fog/5
sh scripts/srun_eval_acc_random_c.sh jpeg   jpeg_template  acc_imagenet_vit16_random ./datasets/data/imagenet-c/jpeg_compression/5

# w/ self-infered z
sh scripts/srun_eval_acc_self_infer.sh vflip             vflip_template_wrong                  vflip_template                0   acc_imagenet_vit16_self_infer_wy
sh scripts/srun_eval_acc_self_infer.sh rotation          rotation_template_wrong               rotation_template             0   acc_imagenet_vit16_self_infer_wy
sh scripts/srun_eval_acc_self_infer.sh elastic_transform elastic_transform_template_wrong      elastic_transform_template    0   acc_imagenet_vit16_self_infer_wy
sh scripts/srun_eval_acc_self_infer.sh invert            invert_template_wrong                 invert_template               0   acc_imagenet_vit16_self_infer_wy
sh scripts/srun_eval_acc_self_infer.sh solarize          solarize_template_wrong               solarize_template             0   acc_imagenet_vit16_self_infer_wy
sh scripts/srun_eval_acc_self_infer.sh blur              blur_template_wrong                   blur_template                 0   acc_imagenet_vit16_self_infer_wy
sh scripts/srun_eval_acc_self_infer.sh grayscale         grayscale_template_wrong              grayscale_template            0   acc_imagenet_vit16_self_infer_wy
sh scripts/srun_eval_acc_self_infer.sh bright            bright_template_wrong                 bright_template               0   acc_imagenet_vit16_self_infer_wy

sh scripts/srun_eval_acc_self_infer_c.sh noise  noise_template_wrong  noise_template  0 acc_imagenet_vit16_self_infer_wy ./datasets/data/imagenet-c/shot_noise/5
sh scripts/srun_eval_acc_self_infer_c.sh snow   snow_template_wrong   snow_template   0 acc_imagenet_vit16_self_infer_wy ./datasets/data/imagenet-c/snow/5
sh scripts/srun_eval_acc_self_infer_c.sh frost  frost_template_wrong  frost_template  0 acc_imagenet_vit16_self_infer_wy ./datasets/data/imagenet-c/frost/5
sh scripts/srun_eval_acc_self_infer_c.sh fog    fog_template_wrong    fog_template    0 acc_imagenet_vit16_self_infer_wy ./datasets/data/imagenet-c/fog/5
sh scripts/srun_eval_acc_self_infer_c.sh jpeg   jpeg_template_wrong   jpeg_template   0 acc_imagenet_vit16_self_infer_wy ./datasets/data/imagenet-c/jpeg_compression/5


###################### CLIP can infer contextual attributes (Table 3) ######################
# method 1: w/ y
sh scripts/srun_infer_z.sh vflip             vflip_template_wrong                  vflip_template                0   infer_z_imagenet_vit16_wy
sh scripts/srun_infer_z.sh rotation          rotation_template_wrong               rotation_template             0   infer_z_imagenet_vit16_wy
sh scripts/srun_infer_z.sh elastic_transform elastic_transform_template_wrong      elastic_transform_template    0   infer_z_imagenet_vit16_wy
sh scripts/srun_infer_z.sh invert            invert_template_wrong                 invert_template               0   infer_z_imagenet_vit16_wy
sh scripts/srun_infer_z.sh solarize          solarize_template_wrong               solarize_template             0   infer_z_imagenet_vit16_wy
sh scripts/srun_infer_z.sh blur              blur_template_wrong                   blur_template                 0   infer_z_imagenet_vit16_wy
sh scripts/srun_infer_z.sh grayscale         grayscale_template_wrong              grayscale_template            0   infer_z_imagenet_vit16_wy
sh scripts/srun_infer_z.sh bright            bright_template_wrong                 bright_template               0   infer_z_imagenet_vit16_wy

sh scripts/srun_infer_z_c.sh noise  noise_template_wrong  noise_template  0 infer_z_imagenet_vit16_wy ./datasets/data/imagenet-c/shot_noise/5
sh scripts/srun_infer_z_c.sh snow   snow_template_wrong   snow_template   0 infer_z_imagenet_vit16_wy ./datasets/data/imagenet-c/snow/5
sh scripts/srun_infer_z_c.sh frost  frost_template_wrong  frost_template  0 infer_z_imagenet_vit16_wy ./datasets/data/imagenet-c/frost/5
sh scripts/srun_infer_z_c.sh fog    fog_template_wrong    fog_template    0 infer_z_imagenet_vit16_wy ./datasets/data/imagenet-c/fog/5
sh scripts/srun_infer_z_c.sh jpeg   jpeg_template_wrong   jpeg_template   0 infer_z_imagenet_vit16_wy ./datasets/data/imagenet-c/jpeg_compression/5

# method 2: w/o y

sh scripts/srun_infer_z.sh vflip             vflip_template_wrong                  vflip_template                1   infer_z_imagenet_vit16_woy
sh scripts/srun_infer_z.sh rotation          rotation_template_wrong               rotation_template             1   infer_z_imagenet_vit16_woy
sh scripts/srun_infer_z.sh elastic_transform elastic_transform_template_wrong      elastic_transform_template    1   infer_z_imagenet_vit16_woy
sh scripts/srun_infer_z.sh invert            invert_template_wrong                 invert_template               1   infer_z_imagenet_vit16_woy
sh scripts/srun_infer_z.sh solarize          solarize_template_wrong               solarize_template             1   infer_z_imagenet_vit16_woy
sh scripts/srun_infer_z.sh blur              blur_template_wrong                   blur_template                 1   infer_z_imagenet_vit16_woy
sh scripts/srun_infer_z.sh grayscale         grayscale_template_wrong              grayscale_template            1   infer_z_imagenet_vit16_woy
sh scripts/srun_infer_z.sh bright            bright_template_wrong                 bright_template               1   infer_z_imagenet_vit16_woy

sh scripts/srun_infer_z_c.sh noise  noise_template_wrong  noise_template  1 infer_z_imagenet_vit16_woy ./datasets/data/imagenet-c/shot_noise/5
sh scripts/srun_infer_z_c.sh snow   snow_template_wrong   snow_template   1 infer_z_imagenet_vit16_woy ./datasets/data/imagenet-c/snow/5
sh scripts/srun_infer_z_c.sh frost  frost_template_wrong  frost_template  1 infer_z_imagenet_vit16_woy ./datasets/data/imagenet-c/frost/5
sh scripts/srun_infer_z_c.sh fog    fog_template_wrong    fog_template    1 infer_z_imagenet_vit16_woy ./datasets/data/imagenet-c/fog/5
sh scripts/srun_infer_z_c.sh jpeg   jpeg_template_wrong   jpeg_template   1 infer_z_imagenet_vit16_woy ./datasets/data/imagenet-c/jpeg_compression/5

###################### PerceptionCLIP improves zero-shot generalization on ImageNet (Table 4) ######################
# Here, we only show scripts for ImageNet. You can change the dataset name to ImageNetV2, ImageNetA, ImageNetR, ImageNetSketch, and change the composition of attributes.
# single attribute
sh scripts/srun_twostep_compose.sh ImageNet ViT-B/16 imagenet_main_template imagenet_factor_templates orientation 0 object imagnet_ours_wy_vit16
sh scripts/srun_twostep_compose.sh ImageNet ViT-B/16 imagenet_main_template imagenet_factor_templates background 0 object imagnet_ours_wy_vit16
sh scripts/srun_twostep_compose.sh ImageNet ViT-B/16 imagenet_main_template imagenet_factor_templates quality 0 object imagnet_ours_wy_vit16
sh scripts/srun_twostep_compose.sh ImageNet ViT-B/16 imagenet_main_template imagenet_factor_templates illumination 0 object imagnet_ours_wy_vit16
sh scripts/srun_twostep_compose.sh ImageNet ViT-B/16 imagenet_main_template imagenet_factor_templates quantity 0 object imagnet_ours_wy_vit16
sh scripts/srun_twostep_compose.sh ImageNet ViT-B/16 imagenet_main_template imagenet_factor_templates perspective 0 object imagnet_ours_wy_vit16
sh scripts/srun_twostep_compose.sh ImageNet ViT-B/16 imagenet_main_template imagenet_factor_templates art 0 object imagnet_ours_wy_vit16
sh scripts/srun_twostep_compose.sh ImageNet ViT-B/16 imagenet_main_template imagenet_factor_templates medium 0 object imagnet_ours_wy_vit16
sh scripts/srun_twostep_compose.sh ImageNet ViT-B/16 imagenet_main_template imagenet_factor_templates condition 0 object imagnet_ours_wy_vit16
sh scripts/srun_twostep_compose.sh ImageNet ViT-B/16 imagenet_main_template imagenet_factor_templates color-scheme 0 object imagnet_ours_wy_vit16
sh scripts/srun_twostep_compose.sh ImageNet ViT-B/16 imagenet_main_template imagenet_factor_templates tool 0 object imagnet_ours_wy_vit16

# composition of attributes
sh scripts/srun_twostep_compose.sh ImageNet ViT-B/16 imagenet_main_template imagenet_factor_templates quality,condition 0 object imagnet_ours_wy_vit16
sh scripts/srun_twostep_compose.sh ImageNet ViT-B/16 imagenet_main_template imagenet_factor_templates background,quality,condition 0 object imagnet_ours_wy_vit16
sh scripts/srun_twostep_compose.sh ImageNet ViT-B/16 imagenet_main_template imagenet_factor_templates perspective,background,quality,condition 0 object imagnet_ours_wy_vit16

###################### PerceptionCLIP improves zero-shot generalization on domain datasets (Table 5) ######################
# For every dataset, we measure:
# simple template
# domain template
# domain template + contextual attributes

sh scripts/srun_onestep.sh        CUB200    simple_template         ViT-B/16 cub200_simple_vit16
sh scripts/srun_onestep.sh        CUB200    cub200_simple_template  ViT-B/16 cub200_simple_vit16
sh scripts/srun_twostep_compose.sh CUB200 ViT-B/16 cub200_main_template cub200_factor_templates size,background,condition 0 bird cub200_ours_wy_vit16

sh scripts/srun_onestep.sh        EuroSAT    simple_template          ViT-B/16 eurosat_simple_vit16
sh scripts/srun_onestep.sh        EuroSAT    eurosat_simple_template  ViT-B/16 eurosat_simple_vit16
sh scripts/srun_twostep_compose.sh EuroSAT ViT-B/16 eurosat_main_template eurosat_factor_templates condition,source        0  place eurosat_ours_wy_vit16

sh scripts/srun_onestep.sh        Places365    simple_template          ViT-B/16 places_simple_vit16
sh scripts/srun_onestep.sh        Places365    places_simple_template   ViT-B/16 places_simple_vit16
sh scripts/srun_twostep_compose.sh Places365 ViT-B/16 places_main_template places_factor_templates background,quality,condition      0  place places_ours_wy_vit16

sh scripts/srun_onestep.sh        Flowers102    simple_template              ViT-B/16 flowers_simple_vit16
sh scripts/srun_onestep.sh        Flowers102    flowers102_simple_template   ViT-B/16 flowers_simple_vit16
sh scripts/srun_twostep_compose.sh Flowers102 ViT-B/16 flowers102_main_template flowers102_factor_templates background,illumination,quality,condition    0 flower flowers_ours_wy_vit16

sh scripts/srun_onestep.sh        Food101    simple_template        ViT-B/16 food_simple_vit16
sh scripts/srun_onestep.sh        Food101    food_simple_template   ViT-B/16 food_simple_vit16
sh scripts/srun_twostep_compose.sh Food101 ViT-B/16 food_main_template food_factor_templates cuisines,condition     0  food food_ours_wy_vit16

sh scripts/srun_onestep.sh        OxfordPets    simple_template              ViT-B/16 oxfordpets_simple_vit16
sh scripts/srun_onestep.sh        OxfordPets    oxfordpets_simple_template   ViT-B/16 oxfordpets_simple_vit16
sh scripts/srun_twostep_compose.sh OxfordPets ViT-B/16 oxfordpets_main_template oxfordpets_factor_templates species,background,pose,interaction         0  pet oxfordpets_ours_wy_vit16


###################### PerceptionCLIP improves group robustness on Waterbirds (Table 7) ######################
# CLIP
sh scripts/srun_waterbirds_ours_onestep.sh waterbirds_simple_template  ViT-B/32  waterbirds_simple_vit32
sh scripts/srun_waterbirds_ours_onestep.sh waterbirds_simple_template  ViT-B/16  waterbirds_simple_vit16
sh scripts/srun_waterbirds_ours_onestep.sh waterbirds_simple_template  RN50      waterbirds_simple_RN50
sh scripts/srun_waterbirds_ours_onestep.sh waterbirds_simple_template  ViT-L/14  waterbirds_simple_vit14

# simple background
sh scripts/srun_waterbirds_ours_twostep.sh waterbirds_background_template  ViT-B/32  0  waterbirds_ours_wy_vit32
sh scripts/srun_waterbirds_ours_twostep.sh waterbirds_background_template  ViT-B/16  0  waterbirds_ours_wy_vit16
sh scripts/srun_waterbirds_ours_twostep.sh waterbirds_background_template  RN50      0  waterbirds_ours_wy_RN50
sh scripts/srun_waterbirds_ours_twostep.sh waterbirds_background_template  ViT-L/14      0  waterbirds_ours_wy_vit14

sh scripts/srun_waterbirds_ours_twostep.sh waterbirds_background_template  ViT-B/32  1  waterbirds_ours_woy_vit32
sh scripts/srun_waterbirds_ours_twostep.sh waterbirds_background_template  ViT-B/16  1  waterbirds_ours_woy_vit16
sh scripts/srun_waterbirds_ours_twostep.sh waterbirds_background_template  RN50      1  waterbirds_ours_woy_RN50
sh scripts/srun_waterbirds_ours_twostep.sh waterbirds_background_template  ViT-L/14      1  waterbirds_ours_woy_vit14

# complex background
sh scripts/srun_waterbirds_ours_twostep_compose.sh background  ViT-B/32  0  waterbirds_ours_wy_vit32_factor
sh scripts/srun_waterbirds_ours_twostep_compose.sh background  ViT-B/16  0  waterbirds_ours_wy_vit16_factor
sh scripts/srun_waterbirds_ours_twostep_compose.sh background  RN50      0   waterbirds_ours_wy_RN50_factor
sh scripts/srun_waterbirds_ours_twostep_compose.sh background  ViT-L/14  0  waterbirds_ours_wy_vit14_factor

sh scripts/srun_waterbirds_ours_twostep_compose.sh background  ViT-B/32  1  waterbirds_ours_woy_vit32_factor
sh scripts/srun_waterbirds_ours_twostep_compose.sh background  ViT-B/16  1  waterbirds_ours_woy_vit16_factor
sh scripts/srun_waterbirds_ours_twostep_compose.sh background  RN50      1   waterbirds_ours_woy_RN50_factor
sh scripts/srun_waterbirds_ours_twostep_compose.sh background  ViT-L/14  1  waterbirds_ours_woy_vit14_factor

###################### PerceptionCLIP improves group robustness on CelebA (Table 8) ######################

# CLIP
sh scripts/srun_celeba_ours_onestep.sh celeba_simple_template  ViT-B/32  celeba_simple_vit32
sh scripts/srun_celeba_ours_onestep.sh celeba_simple_template  ViT-B/16  celeba_simple_vit16
sh scripts/srun_celeba_ours_onestep.sh celeba_simple_template  RN50      celeba_simple_RN50
sh scripts/srun_celeba_ours_onestep.sh celeba_simple_template  ViT-L/14  celeba_simple_vit14

# PerceptionCLIP w/ gender
sh scripts/srun_celeba_ours_twostep.sh celeba_gender_template  ViT-B/32  0  celeba_ours_wy_vit32
sh scripts/srun_celeba_ours_twostep.sh celeba_gender_template  ViT-B/16  0  celeba_ours_wy_vit16
sh scripts/srun_celeba_ours_twostep.sh celeba_gender_template  RN50      0  celeba_ours_wy_RN50
sh scripts/srun_celeba_ours_twostep.sh celeba_gender_template  ViT-L/14  0  celeba_ours_wy_vit14

sh scripts/srun_celeba_ours_twostep.sh celeba_gender_template  ViT-B/32  1  celeba_ours_woy_vit32
sh scripts/srun_celeba_ours_twostep.sh celeba_gender_template  ViT-B/16  1  celeba_ours_woy_vit16
sh scripts/srun_celeba_ours_twostep.sh celeba_gender_template  RN50      1  celeba_ours_woy_RN50
sh scripts/srun_celeba_ours_twostep.sh celeba_gender_template  ViT-L/14  1  celeba_ours_woy_vit14

# PerceptionCLIP w/ gender + age
sh scripts/srun_celeba_ours_twostep_compose.sh gender,age  ViT-B/32  0  celeba_ours_wy_vit32_factor
sh scripts/srun_celeba_ours_twostep_compose.sh gender,age  ViT-B/16  0  celeba_ours_wy_vit16_factor
sh scripts/srun_celeba_ours_twostep_compose.sh gender,age  RN50      0   celeba_ours_wy_RN50_factor
sh scripts/srun_celeba_ours_twostep_compose.sh gender,age  ViT-L/14  0  celeba_ours_wy_vit14_factor

sh scripts/srun_celeba_ours_twostep_compose.sh gender,age  ViT-B/32  1  celeba_ours_woy_vit32_factor
sh scripts/srun_celeba_ours_twostep_compose.sh gender,age  ViT-B/16  1  celeba_ours_woy_vit16_factor
sh scripts/srun_celeba_ours_twostep_compose.sh gender,age  RN50      1   celeba_ours_woy_RN50_factor
sh scripts/srun_celeba_ours_twostep_compose.sh gender,age  ViT-L/14  1  celeba_ours_woy_vit14_factor

# PerceptionCLIP w/ gender + age + race
sh scripts/srun_celeba_ours_twostep_compose.sh gender,age,race  ViT-B/32  0  celeba_ours_wy_vit32_factor
sh scripts/srun_celeba_ours_twostep_compose.sh gender,age,race  ViT-B/16  0  celeba_ours_wy_vit16_factor
sh scripts/srun_celeba_ours_twostep_compose.sh gender,age,race  RN50      0   celeba_ours_wy_RN50_factor
sh scripts/srun_celeba_ours_twostep_compose.sh gender,age,race  ViT-L/14  0  celeba_ours_wy_vit14_factor


sh scripts/srun_celeba_ours_twostep_compose.sh gender,age,race  ViT-B/32  1  celeba_ours_woy_vit32_factor
sh scripts/srun_celeba_ours_twostep_compose.sh gender,age,race  ViT-B/16  1  celeba_ours_woy_vit16_factor
sh scripts/srun_celeba_ours_twostep_compose.sh gender,age,race  RN50      1   celeba_ours_woy_RN50_factor
sh scripts/srun_celeba_ours_twostep_compose.sh gender,age,race  ViT-L/14  1  celeba_ours_woy_vit14_factor
