import torch
import argparse
import csv
import os
import src.datasets as datasets
import src.templates as templates
from src.datasets.common import get_dataloader, maybe_dictionarize
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
        default='./results/zero_shot_inference/eval_acc_ours',
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.,
        help="the temperature used for intervene p(z|x)",
    )
    parser.add_argument(
        "--num_attrs",
        type=int,
        default=2,
        help="number of attribute values",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="number of label values",
    )
    parser.add_argument(
        "--eval_group",
        type=bool,
        default=False,
        help="Evaluate group robustness",
    )
    parser.add_argument(
        "--factors",
        default=None,
        type=lambda x: x.split(","),
    )
    parser.add_argument(
        "--main_template",
        type=str,
        default="main_template",
        help="text prompt for Y",
    )
    parser.add_argument(
        "--factor_templates",
        type=str,
        default="factor_templates",
        help="text prompt for Zs",
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
                            num_workers=args.workers)
    print(f"Eval dataset: {args.dataset}")

    # load template
    if args.factors is not None:
        main_template = getattr(templates, args.main_template)
        factor_templates = getattr(templates, args.factor_templates)

        factor_templates = generate_composite_factors(factor_templates,
                                                      selected_factors=args.factors)
        template_list = compose_template(main_template, factor_templates)
    else:
        raise Exception('Please provide factor names!')

    # create classifier
    if args.infer_mode == 0:
        # w/ y
        args.num_factor_value = len(template_list)
        classification_head = get_zeroshot_classifier_flat_advance(args, model.model,
                                                                   dataset.classnames,
                                                                   template_list)
        factor_head = None

    elif args.infer_mode == 1:
        # w/o y
        template_list_woy = template_convert(template_list, args.convert_text)
        args.num_factor_value = len(template_list)
        factor_head = get_zeroshot_classifier_puretext_advance(args, model.model,
                                                               template_list_woy)
        classification_head = get_zeroshot_classifier_flat_advance(args, model.model,
                                                                   dataset.classnames,
                                                                   template_list)
        factor_head = factor_head.cuda()

    classification_head = classification_head.cuda()

    if args.eval_group:
        acc, worst, _ = classify(model, classification_head, dataset, args, factor_head)
    else:
        acc = classify(model, classification_head, dataset, args, factor_head)

    # save result to csv
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    file_path = os.path.join(args.save_path, args.save_name + '.csv')
    with open(file_path, mode='a') as file:
        writer = csv.writer(file)
        if args.eval_augmentation_2 != "None":
            writer.writerow(
                [args.eval_augmentation] + [args.eval_augmentation_2] + [args.temperature] + [acc])
        elif args.eval_augmentation != "None":
            writer.writerow([args.eval_augmentation] + [args.temperature] + [acc])
        else:
            if args.eval_group:
                if args.factors is not None:
                    writer.writerow([args.factors] + [args.temperature] + [acc] + [worst])
                else:
                    writer.writerow([args.template] + [args.temperature] + [acc] + [worst])
            else:
                if args.factors is not None:
                    writer.writerow([args.factors] + [args.temperature] + [acc])
                else:
                    writer.writerow([args.template] + [args.temperature] + [acc])

    print('results saved!')

    return


def classify(model, classification_head, dataset, args, factor_head=None):
    model.eval()
    classification_head.eval()
    dataloader = get_dataloader(dataset,
                                is_train=args.eval_trainset,
                                args=args,
                                image_encoder=None)
    batched_data = enumerate(dataloader)
    device = args.device

    if args.eval_group:
        group_correct = torch.zeros((args.num_attrs, args.num_labels))
        group_cnt = torch.zeros((args.num_attrs, args.num_labels))
    else:
        top1, correct, n = 0., 0., 0.

    with torch.no_grad():

        for i, data in batched_data:
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)
            if args.eval_group:
                a = data['metadata'].to(device)

            # infer z
            if args.infer_mode == 0:
                # w/ y
                # get flat predictions f(y, z ; x)
                logits = utils.get_logits(x, model, classification_head)

                # P(z|x) = \frac{\sum_y{exp(f(y,z;x)/t)}}{\sum_z\sum_y{exp(f(y,z;x)/t)}}
                factor_probs = (logits / args.temperature).exp()
                factor_probs = factor_probs.view(factor_probs.shape[0], -1,
                                                 args.num_factor_value)  # N x |Y| x |Z|
                factor_probs = factor_probs.sum(dim=1)  # \sum_y{exp(f(y,z;x)/t)}
                factor_probs = factor_probs / factor_probs.sum(dim=1).unsqueeze(1)

            elif args.infer_mode == 1:
                # w/o y
                # P(z|x) = \frac{exp(f(z;x)/t)}{\sum_z{exp(f(z;x)/t)}}
                factor_logits = utils.get_logits(x, model, factor_head)
                factor_probs = factor_logits / args.temperature
                factor_probs = torch.softmax(factor_probs, dim=1)

                # get flat predictions f(y, z ; x)
                logits = utils.get_logits(x, model, classification_head)

            # infer y conditioned on x and z
            # P(y|x,z) = \frac{exp(f(y,z;x)}{\sum_y{exp(f(y,z;x))}}
            probs = (logits / 1).exp()
            probs = probs.view(probs.shape[0], -1, args.num_factor_value)
            probs = probs / probs.sum(dim=1).unsqueeze(1)

            # P(y|x) = \sum_z{P(z|x) * P(y|x,z)}
            probs = probs * factor_probs.unsqueeze(1)
            probs = probs.sum(dim=2)

            pred = probs.argmax(dim=1, keepdim=True).to(device)

            if args.eval_group:
                batch_group_correct, batch_group_cnt = group_accuracy(args, pred, y, a)
                group_correct += batch_group_correct
                group_cnt += batch_group_cnt
            else:
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

    if args.eval_group:
        acc = group_correct.sum() / group_cnt.sum() * 100
        group_acc = torch.nan_to_num(group_correct / group_cnt) * 100
        worst_acc = group_acc.min()
        print(
            f"Avg Accuracy: {acc.item():.2f} | Worst Accuracy: {worst_acc.item():.2f} | Group Acc: {group_acc.tolist()}")
        return acc.item(), worst_acc.item(), group_acc.tolist()
    else:
        top1 = correct / n
        print(f"Accuracy: {top1}")
        return top1


if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    main(args)
