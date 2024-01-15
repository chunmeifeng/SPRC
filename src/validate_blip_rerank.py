import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import squarepad_transform, FashionIQDataset, targetpad_transform, CIRRDataset
from combiner import Combiner
from utils import extract_index_features, collate_fn, element_wise_sum, device
import os
from pathlib import Path
import shutil

def compute_fiq_val_metrics(relative_val_dataset: FashionIQDataset, blip_model, index_features: torch.tensor,
                            index_names: List[str], txt_processors) -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ dataset
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """

    # Generate predictions
    pred_sim, target_names, reference_names, captions_all, name2feat = generate_fiq_val_predictions(blip_model, relative_val_dataset,
                                                                    index_names, index_features, txt_processors)

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Compute the distances and sort the results
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]


    # re-rank
    i = 0
    step = 40
    top = 40
    while i < len(sorted_index_names):
        if step + i > len(sorted_index_names):
            step = len(sorted_index_names) - i
        reference_name = reference_names[i: i + step]
        caption = captions_all[i: i + step]
        targets_top100 = sorted_index_names[i: i + step, :top]
        if step == 1:
            reference_feats = itemgetter(*reference_name)(name2feat).unsqueeze(0)
        else:
            reference_feats = torch.stack(itemgetter(*reference_name)(name2feat)) 
        target_feats = torch.stack(itemgetter(*targets_top100.reshape(-1))(name2feat)) 
        with torch.no_grad():
            top100_rank = blip_model.inference_rerank(reference_feats, target_feats, caption)
        distances_top100 = 1 - top100_rank
        distances_top100 = distances_top100.reshape(-1, top)
        sorted_indices_top100 = torch.argsort(distances_top100, dim=-1).cpu()
        # change sorted_indices based on re-ranking
        for j in range(step):
            sorted_index_names[i + j, :top] = sorted_index_names[i + j, :top][sorted_indices_top100[j]]
        i = i + step


    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


def generate_fiq_val_predictions(blip_model, relative_val_dataset: FashionIQDataset,
                                 index_names: List[str], index_features, txt_processors) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Compute FashionIQ predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features and target names
    """
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=16,
                                     num_workers=4, pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features[-1]))

    # Initialize predicted features and target names
    target_names = []
    reference_names_all = []
    distance = []
    captions_all = []

    for reference_names, batch_target_names, captions in tqdm(relative_val_loader):  # Load data

        # Concatenate the captions in a deterministic way
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]
        input_captions = [txt_processors["eval"](caption) for caption in input_captions]
        # Compute the predicted features
        with torch.no_grad():
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if len(input_captions) == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_distance = blip_model.inference(reference_image_features, index_features[0], input_captions)
            distance.append(batch_distance)
            captions_all += input_captions

        target_names.extend(batch_target_names)
        reference_names_all.extend(reference_names)
    
    distance = torch.vstack(distance)

    return distance, target_names, reference_names_all, captions_all, name_to_feat


def fashioniq_val_retrieval(dress_type: str, combining_function: callable, clip_model: CLIP, preprocess: callable):
    """
    Perform retrieval on FashionIQ validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param dress_type: FashionIQ category on which perform the retrieval
    :param combining_function:function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_model: CLIP model
    :param preprocess: preprocess pipeline
    """

    clip_model = clip_model.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
    relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)

    return compute_fiq_val_metrics(relative_val_dataset, clip_model, index_features, index_names,
                                   combining_function)


def compute_cirr_val_metrics(relative_val_dataset: CIRRDataset, blip_model, index_features,
                             index_names: List[str], txt_processors) -> Tuple[
    float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR dataset
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """
    # Generate predictions
    pred_sim, reference_names, target_names, group_members, captions_all, name2feat, name_to_feat_tar = \
        generate_cirr_val_predictions(blip_model, relative_val_dataset, index_names, index_features, txt_processors)

    print("Compute CIRR validation metrics")

    # Compute the distances and sort the results
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    
    sorted_index_names = np.load('sorted_index_names.npy')
    
    # re-rank
    i = 0
    step = 15
    top = 200
    while i < len(sorted_index_names):
        if step + i > len(sorted_index_names):
            step = len(sorted_index_names) - i
        reference_name = reference_names[i: i + step]
        caption = captions_all[i: i + step]
        targets_top100 = sorted_index_names[i: i + step, :top]
        if step == 1:
            reference_feats = itemgetter(*reference_name)(name2feat).unsqueeze(0)
        else:
            reference_feats = torch.stack(itemgetter(*reference_name)(name2feat)) 
        target_feats = torch.stack(itemgetter(*targets_top100.reshape(-1))(name_to_feat_tar)) 
        with torch.no_grad():
            top100_rank = blip_model.inference_rerank(reference_feats, target_feats, caption)
        distances_top100 = 1 - top100_rank
        distances_top100 = distances_top100.reshape(-1, top)
        sorted_indices_top100 = torch.argsort(distances_top100, dim=-1).cpu()
        # change sorted_indices based on re-ranking
        for j in range(step):
            sorted_index_names[i + j, :top] = sorted_index_names[i + j, :top][sorted_indices_top100[j]]
        i = i + step
    # re-rank end


    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)
    sorted_index_names_group = sorted_index_names[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    # vis(sorted_index_names_group, group_labels, reference_names, captions_all, target_names)

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50

def vis(sorted_index_names_group, group_labels, reference_name, captions_all, target_name):
    base_path = '/home/fengchunmei/cir/CLIP4Cir/vis_3'
    img_path = '/home/fengchunmei/cir/CLIP4Cir/cirr_dataset/dev'
    wong_count = 0
    for i in range(len(captions_all)):
        # print(i)
        index_name = sorted_index_names_group[i]
        label = group_labels[i]
        caption = captions_all[i]
        reference = reference_name[i]
        tar_curr = target_name[i]
        if (label[2].sum() > 0).item() is True:
            path_curr: Path = Path(f"{base_path}/{caption}")
            path_curr.mkdir(exist_ok=True, parents=True)
            # copy refernce image
            ref_path = os.path.join(img_path, f'{reference}.png')
            shutil.copy2(ref_path, os.path.join(str(path_curr),  "ref.png"))
            for j in range(len(index_name)):
                idx_j = index_name[j]
                if (label[j].sum() > 0).item() is True:
                    if idx_j != tar_curr:
                        print('wong!')
                        wong_count+=1
                tar_path = os.path.join(img_path, f'{idx_j}.png')
                shutil.copy2(tar_path, os.path.join(str(path_curr),  f"{j}.png"))
    print(wong_count)
    print('finsh vis')



def compute_cirr_val_metrics_relative(relative_val_dataset: CIRRDataset, clip_model: CLIP, index_features: torch.tensor,
                             index_names: List[str], combining_function: callable) -> Tuple[
    float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR dataset
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """
    # Generate predictions
    predicted_features, relative_pred, reference_names, target_names, group_members = \
        generate_cirr_val_predictions_relative(clip_model, relative_val_dataset, combining_function, index_names, index_features)

    print("Compute CIRR validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    distances2 = 1 - relative_pred
    distances3 = distances * 0.8 + distances2 * 0.2

    _, _, _, recall_at1, recall_at5, recall_at10, recall_at50 = get_results(distances,index_names, reference_names, target_names, group_members)
    print(f'normal: recall_at1:{recall_at1:.2f},recall_at5:{recall_at5:.2f},recall_at10:{recall_at10:.2f},recall_at50:{recall_at50:.2f},')
    _, _, _, recall_at1, recall_at5, recall_at10, recall_at50 = get_results(distances2,index_names, reference_names, target_names, group_members)
    print(f'relative: recall_at1:{recall_at1:.2f},recall_at5:{recall_at5:.2f},recall_at10:{recall_at10:.2f},recall_at50:{recall_at50:.2f},')

    group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = get_results(distances3,index_names, reference_names, target_names, group_members)

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50

def get_results(distances,index_names, reference_names, target_names, group_members):
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


def generate_cirr_val_predictions(blip_model, relative_val_dataset: CIRRDataset, 
                                  index_names: List[str], index_features, txt_processors) -> \
        Tuple[torch.tensor, List[str], List[str], List[List[str]]]:
    """
    Compute CIRR predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features, reference names, target names and group members
    """
    print("Compute CIRR validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=2,
                                     pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features[1]))
    name_to_feat_tar = dict(zip(index_names, index_features[0]))

    # Initialize predicted features, target_names, group_members and reference_names
    distance = []
    target_names = []
    group_members = []
    reference_names = []
    captions_all = []

    for batch_reference_names, batch_target_names, captions, batch_group_members in tqdm(
            relative_val_loader):  # Load data
        batch_group_members = np.array(batch_group_members).T.tolist()
        captions = [txt_processors["eval"](caption) for caption in captions]
        # Compute the predicted features
        with torch.no_grad():
            # text_features = clip_model.encode_text(text_inputs)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if len(captions) == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_distance = blip_model.inference(reference_image_features, index_features[0], captions)
            distance.append(batch_distance)
            captions_all += captions

        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)
    
    distance = torch.vstack(distance)

    return distance, reference_names, target_names, group_members, captions_all, name_to_feat, name_to_feat_tar


def generate_cirr_val_predictions_relative(clip_model: CLIP, relative_val_dataset: CIRRDataset,
                                  combining_function: callable, index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[str], List[List[str]]]:
    """
    Compute CIRR predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features, reference names, target names and group members
    """
    print("Compute CIRR validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=2,
                                     pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device, non_blocking=True)
    ref_feats = []
    text_feats = []
    target_names = []
    relative_all = []
    group_members = []
    reference_names = []

    for batch_reference_names, batch_target_names, captions, batch_group_members in tqdm(
            relative_val_loader):  # Load data
        text_inputs = clip.tokenize(captions).to(device, non_blocking=True)
        batch_group_members = np.array(batch_group_members).T.tolist()

        # Compute the predicted features
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)

            relative_feats = F.normalize(index_features.unsqueeze(0) - reference_image_features.unsqueeze(1), dim=-1)
        
            pred_relative = torch.matmul(relative_feats, F.normalize(text_features, dim=-1).unsqueeze(-1)).squeeze(-1)

        relative_all.append(pred_relative)
        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))

        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)

    relative_all = torch.vstack(relative_all)

    return predicted_features, relative_all, reference_names, target_names, group_members


def cirr_val_retrieval(combining_function: callable, clip_model: CLIP, preprocess: callable):
    """
    Perform retrieval on CIRR validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_model: CLIP model
    :param preprocess: preprocess pipeline
    """

    clip_model = clip_model.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)

    return compute_cirr_val_metrics(relative_val_dataset, clip_model, index_features, index_names,
                                    combining_function)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--combining-function", type=str, required=True,
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--combiner-path", type=Path, help="path to trained Combiner")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")

    args = parser.parse_args()

    clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    if args.clip_model_path:
        print('Trying to load the CLIP model')
        saved_state_dict = torch.load(args.clip_model_path, map_location=device)
        clip_model.load_state_dict(saved_state_dict["CLIP"])
        print('CLIP model loaded successfully')

    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.transform == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        print('CLIP default preprocess pipeline is used')
        preprocess = clip_preprocess

    if args.combining_function.lower() == 'sum':
        if args.combiner_path:
            print("Be careful, you are using the element-wise sum as combining_function but you have also passed a path"
                  " to a trained Combiner. Such Combiner will not be used")
        combining_function = element_wise_sum
    elif args.combining_function.lower() == 'combiner':
        combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device, non_blocking=True)
        state_dict = torch.load(args.combiner_path, map_location=device)
        combiner.load_state_dict(state_dict["Combiner"])
        combiner.eval()
        combining_function = combiner.combine_features
    else:
        raise ValueError("combiner_path should be in ['sum', 'combiner']")

    if args.dataset.lower() == 'cirr':
        group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = \
            cirr_val_retrieval(combining_function, clip_model, preprocess)

        print(f"{group_recall_at1 = }")
        print(f"{group_recall_at2 = }")
        print(f"{group_recall_at3 = }")
        print(f"{recall_at1 = }")
        print(f"{recall_at5 = }")
        print(f"{recall_at10 = }")
        print(f"{recall_at50 = }")

    elif args.dataset.lower() == 'fashioniq':
        average_recall10_list = []
        average_recall50_list = []

        shirt_recallat10, shirt_recallat50 = fashioniq_val_retrieval('shirt', combining_function, clip_model,
                                                                     preprocess)
        average_recall10_list.append(shirt_recallat10)
        average_recall50_list.append(shirt_recallat50)

        dress_recallat10, dress_recallat50 = fashioniq_val_retrieval('dress', combining_function, clip_model,
                                                                     preprocess)
        average_recall10_list.append(dress_recallat10)
        average_recall50_list.append(dress_recallat50)

        toptee_recallat10, toptee_recallat50 = fashioniq_val_retrieval('toptee', combining_function, clip_model,
                                                                       preprocess)
        average_recall10_list.append(toptee_recallat10)
        average_recall50_list.append(toptee_recallat50)

        print(f"\n{shirt_recallat10 = }")
        print(f"{shirt_recallat50 = }")

        print(f"{dress_recallat10 = }")
        print(f"{dress_recallat50 = }")

        print(f"{toptee_recallat10 = }")
        print(f"{toptee_recallat50 = }")

        print(f"average recall10 = {mean(average_recall10_list)}")
        print(f"average recall50 = {mean(average_recall50_list)}")
    else:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")


if __name__ == '__main__':
    main()
