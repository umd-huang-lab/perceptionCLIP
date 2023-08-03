import torch
import argparse
import os
import csv
import src.datasets as datasets
import src.templates as templates
from src.datasets.common import get_dataloader, maybe_dictionarize
import src.models.transformer as transformer
from src.zero_shot_inference.utils import *
from src.models import utils
from src.models.modeling import CLIPEncoder


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_location",
        type=str,
        default="./datasets/data/",
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--dataset",
        default='ImageNet',
        type=str,
        help=
        "Which datasets to use for evaluation",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="simple_template",
        help=
        "Which prompt template is used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        help="The type of model (e.g. RN50, ViT-B/32).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument("--workers",
                        type=int,
                        default=2,
                        help="Number of dataloader workers per GPU.")
    parser.add_argument(
        "--eval_augmentation",
        type=str,
        default="None",
        help="The type of data augmentation used for evaluation",
    )
    parser.add_argument(
        "--eval_augmentation_2",
        type=str,
        default="None",
        help="The type of data augmentation used for evaluation",
    )
    parser.add_argument(
        "--eval_augmentation_param",
        type=int,
        default=1,
        help="The parameter of data augmentation used for evaluation",
    )
    parser.add_argument(
        "--eval_trainset",
        type=bool,
        default=False,
        help="Evaluate on training set or test set",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default='tmp',
        help="Name of the csv",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default='./results/zero_shot_inference/eval_sim',
        help="path to save the csv",
    )
    parser.add_argument(
        "--finetuned_checkpoint",
        type=str,
        default=None,
        help="model_path",
    )
    parser.add_argument(
        "--checkpoint_mode",
        type=int,
        default=0,
        help="mode of checkpoint",
    )
    parser.add_argument(
        "--random_descrip",
        type=bool,
        default=False,
        help="randomize the description of the factors or not",
    )

    parsed_args = parser.parse_args()

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return parsed_args


def evaluate_sim(model, classification_head, dataset, args):
    model.eval()
    classification_head.eval()

    dataloader = get_dataloader(dataset,
                                is_train=args.eval_trainset,
                                args=args,
                                image_encoder=None)
    batched_data = enumerate(dataloader)
    device = args.device

    with torch.no_grad():
        sims, n = 0., 0.

        for i, data in batched_data:
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            if args.eval_augmentation != "None":
                x = transformer.augment_image(args.eval_augmentation, x,
                                              args.eval_augmentation_param)
            if args.eval_augmentation_2 != "None":
                x = transformer.augment_image(args.eval_augmentation_2, x,
                                              args.eval_augmentation_param)

            logits = utils.get_logits(x, model, classification_head)
            sim = logits.gather(1, torch.unsqueeze(y, 1))
            sims += sum(sim).item()
            n += x.shape[0]

    return sims / n


def main(args):
    # load model
    model = CLIPEncoder(args, keep_lang=True)
    print(f"Model arch: {args.model}")
    if args.finetuned_checkpoint is not None:
        if args.checkpoint_mode == 0:
            finetuned_checkpoint = torch.load(args.finetuned_checkpoint)
            model.load_state_dict(finetuned_checkpoint['model_state_dict'])
        elif args.checkpoint_mode == 1:
            model.load(args.finetuned_checkpoint)
        print('finetuned model loaded.')
    model = model.cuda()

    # load data
    dataset_class = getattr(datasets, args.dataset)
    dataset = dataset_class(model.val_preprocess,
                            location=args.data_location,
                            batch_size=args.batch_size,
                            num_workers=args.workers)
    print(f"Eval dataset: {args.dataset}")

    # get template
    assert args.template is not None
    template = getattr(templates, args.template)
    if args.random_descrip:
        template = randomize_template(template)

    # create classifier
    classification_head = get_zeroshot_classifier(args, model.model, dataset.classnames, template)
    classification_head = classification_head.cuda()

    # eval similarity
    sims = evaluate_sim(model, classification_head, dataset, args)
    print(f"Avg similarity: {sims}")

    # save results
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    file_path = os.path.join(args.save_path, args.save_name + '.csv')
    with open(file_path, mode='a') as file:
        writer = csv.writer(file)
        if args.eval_augmentation_2 != "None":
            writer.writerow([args.eval_augmentation] + [args.eval_augmentation_2] + [sims])
        else:
            writer.writerow([args.eval_augmentation] + [sims])
    print('results saved!')

    return


if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    main(args)
