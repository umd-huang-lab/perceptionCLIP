import torch
import argparse
import csv
import os
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
        "--template0",
        type=str,
        default="simple_template",
        help=
        "Which prompt template is used.",
    )
    parser.add_argument(
        "--template1",
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
        default='./results/zero_shot_inference/eval_infer_z',
        help="Name of the csv",
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
        "--infer_mode",
        type=int,
        choices=[0, 1],
        default=0,
        help="method of inferring latent factor. 0: w/ y. 1: w/o y",
    )
    parser.add_argument(
        "--convert_text",
        type=str,
        default='object',
        help="convert template from a list of function to a list of pure text. only use it for infer_mode=1",
    )

    parsed_args = parser.parse_args()

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return parsed_args


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
                            num_workers=args.workers)  # class_descriptions=args.class_descriptions)
    print(f"Eval dataset: {args.dataset}")

    # load imagenet
    dataset_class_imagenet = getattr(datasets, "ImageNet")
    dataset_imagenet = dataset_class_imagenet(model.val_preprocess,
                                              location="./datasets/data/",
                                              batch_size=args.batch_size,
                                              num_workers=args.workers)  # class_descriptions=args.class_descriptions)

    # load template
    template_0 = getattr(templates, args.template0)
    template_1 = getattr(templates, args.template1)
    if args.infer_mode == 1:
        template_0 = template_convert(template_0, args.convert_text)
        template_1 = template_convert(template_1, args.convert_text)
    template_list = [template_0, template_1]
    args.num_factor_value = len(template_list)
    # print(template_list)

    # create classifier
    if args.infer_mode == 0:
        # w/ y
        classification_head = get_zeroshot_classifier_flat_advance(args, model.model,
                                                                   dataset.classnames,
                                                                   template_list)

    elif args.infer_mode == 1:
        # w/o y
        classification_head = get_zeroshot_classifier_puretext_advance(args, model.model,
                                                                       template_list)

    classification_head = classification_head.cuda()

    acc = infer_factor(model, classification_head, dataset, dataset_imagenet, args)

    # save result to csv
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    file_path = os.path.join(args.save_path, args.save_name + '.csv')
    with open(file_path, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([args.eval_augmentation] + [acc])
    print('results saved!')

    return


def infer_factor(model, classification_head, dataset, dataset_imagenet, args):
    model.eval()
    classification_head.eval()
    dataloader = get_dataloader(dataset,
                                is_train=args.eval_trainset,
                                args=args,
                                image_encoder=None)
    batched_data = enumerate(dataloader)

    dataloader_imagenet = get_dataloader(dataset_imagenet,
                                         is_train=args.eval_trainset,
                                         args=args,
                                         image_encoder=None)
    batched_data_imagenet = enumerate(dataloader_imagenet)

    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for (i, data), (j, data_imagenet) in zip(batched_data, batched_data_imagenet):
            data = maybe_dictionarize(data)
            data_imagenet = maybe_dictionarize(data_imagenet)

            # apply data transformation (change z) to half of the data
            if i % 2 == 1:
                x = data['images'].to(device)
                y = data['labels'].to(device)
                y = torch.ones_like(y)
            else:
                x = data_imagenet['images'].to(device)
                y = data_imagenet['labels'].to(device)
                y = torch.zeros_like(y)

            logits = utils.get_logits(x, model, classification_head)

            if args.infer_mode == 0:
                # w/ y
                probs = (logits / 1).exp()
                probs = probs.view(probs.shape[0], -1, args.num_factor_value)  # N x |Y| x |Z|
                probs = probs.sum(dim=1)  # \sum_y{exp(f(y,z;x))}
                pred = probs.argmax(dim=1, keepdim=True)

            elif args.infer_mode == 1:
                # w/o y
                probs = torch.softmax(logits, dim=1)
                pred = logits.argmax(dim=1, keepdim=True)

            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)
        top1 = correct / n

    print(f"Accuracy of inferring z: {top1}")

    return top1


if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    main(args)
