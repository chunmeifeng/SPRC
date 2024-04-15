import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess

from data_utils import squarepad_transform, FashionIQDataset, targetpad_transform, CIRRDataset
from utils import extract_index_features, collate_fn, extract_index_blip_features, device
import os
from pathlib import Path
import shutil
import cv2
import json
from statistics import mean, geometric_mean, harmonic_mean


def compute_fiq_val_metrics(relative_val_dataset: FashionIQDataset, blip_model, index_features: torch.tensor,
                            index_names: List[str], txt_processors, save_memory=False) -> Tuple[float, float]:
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
    pred_sim, target_names, reference_names, captions_all = generate_fiq_val_predictions(blip_model, relative_val_dataset,
                                                                    index_names, index_features, txt_processors, save_memory)

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Compute the distances and sort the results
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50

def vis_fiq(sorted_index_names, reference_names, captions_all, labels, dress_type):
    base_path = os.path.join(os.getcwd(), f'fiq_main/vis{dress_type}')
    img_path = os.path.join(os.getcwd(), 'fashionIQ_dataset/images')
    for i in range(len(captions_all)):
        index_name = sorted_index_names[i]
        label = labels[i]
        caption = captions_all[i]
        reference = reference_names[i]
        if (label[0].sum() > 0).item() is True: 
            path_curr: Path = Path(f"{base_path}/{caption}")
            path_curr.mkdir(exist_ok=True, parents=True)
            # copy refernce image
            ref_path = os.path.join(img_path, f'{reference}.png')
            shutil.copy2(ref_path, os.path.join(str(path_curr),  "ref.png"))
            for j in range(8):
                idx_j = index_name[j]
                tar_path = os.path.join(img_path, f'{idx_j}.png')
                shutil.copy2(tar_path, os.path.join(str(path_curr),  f"{j}.png"))
    print('vis_fiq')  

def vis_fiq_failure2(sorted_index_names_group, reference_name, captions_all, group_labels, target_name, dress_type):

    img_path = os.path.join(os.getcwd(), 'fashionIQ_dataset/images')

    wong_count = 0

    ranking = torch.argmax(group_labels.long(), dim=-1)
    base_path: Path = Path(f"vis_fiq/{dress_type}")
    base_path.mkdir(exist_ok=True, parents=True)

    for i in range(len(captions_all)):
        index_name = sorted_index_names_group[i]
        label = group_labels[i]
        caption = captions_all[i]
        reference = reference_name[i]
        tar_curr = target_name[i]
        # if (label[rank-1].sum() > 0).item() is True:
        if (label[:10].sum() < 1).item() is True: 
            target_pos = torch.argmax(label.long()).item()
            # path_curr: Path = Path(f"{base_path}/{target_pos}_{caption}")
            # path_curr.mkdir(exist_ok=True, parents=True)
            # copy refernce image
            ref_path_curr = os.path.join(img_path, f'{reference}.png')
            img_curr_list = []
            img_curr_list.append(ref_path_curr)
            target_curr = os.path.join(img_path, f'{tar_curr}.png')
            for j in range(5):
                idx_j = index_name[j]
                tar_path = os.path.join(img_path, f'{idx_j}.png')
                img_curr_list.append(tar_path)
            img_curr_list.append(target_curr)

            img_np_curr_list = []
            for path in img_curr_list:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                resized = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
                img_np_curr_list.append(resized)
            try: 
                img_all = np.concatenate(img_np_curr_list, axis=1)
            except:
                print('hehe')
            # put on text
            img_all = cv2.putText(img_all, f"{ranking[i]}_{caption}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (252, 255, 53), 2, cv2.LINE_AA)
            # save
            cv2.imwrite(f"{base_path}/{i}.png", img_all)

    print(wong_count)
    print('finsh failure vis')

def vis_fiq_other(sorted_index_names, reference_names, captions_all, labels, dress_type, model_type):
    base_path = os.path.join(os.getcwd(), f'fiq_{model_type}/vis{dress_type}')
    img_path = os.path.join(os.getcwd(), 'fashionIQ_dataset/images')
    for i in range(len(captions_all)):
        index_name = sorted_index_names[i]
        label = labels[i]
        caption = captions_all[i]
        reference = reference_names[i]
        if (label[:3].sum() < 1).item() is True: 
            path_curr: Path = Path(f"{base_path}/{caption}")
            path_curr.mkdir(exist_ok=True, parents=True)
            # copy refernce image
            ref_path = os.path.join(img_path, f'{reference}.png')
            shutil.copy2(ref_path, os.path.join(str(path_curr),  "ref.png"))
            for j in range(8):
                idx_j = index_name[j]
                tar_path = os.path.join(img_path, f'{idx_j}.png')
                shutil.copy2(tar_path, os.path.join(str(path_curr),  f"{j}.png"))
    print('vis_fiq_other')  

def generate_fiq_val_predictions(blip_model, relative_val_dataset: FashionIQDataset,
                                 index_names: List[str], index_features, txt_processors, save_memory=False) -> \
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
            feature_curr = index_features[0]
            if save_memory:
                feature_curr = feature_curr.to(blip_model.device)
                reference_image_features = reference_image_features.to(blip_model.device)
            batch_distance = blip_model.inference(reference_image_features, feature_curr, input_captions)
            distance.append(batch_distance)
            captions_all += input_captions

        target_names.extend(batch_target_names)
        reference_names_all.extend(reference_names)
    
    distance = torch.vstack(distance)

    return distance, target_names, reference_names_all, captions_all


def fashioniq_val_retrieval(dress_type: str, combining_function: callable, clip_model, preprocess):
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
    pred_sim, reference_names, target_names, group_members, captions_all = \
        generate_cirr_val_predictions(blip_model, relative_val_dataset, index_names, index_features, txt_processors)

    print("Compute CIRR validation metrics")
    if 'the animal is now standing and by himself' in captions_all or "dev-190-0-img0" in reference_names:
        print('hehe')
    # Compute the distances and sort the results
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    

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

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50




def compute_cirr_val_metrics_relative(relative_val_dataset: CIRRDataset, clip_model, index_features: torch.tensor,
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

    return distance, reference_names, target_names, group_members, captions_all


def generate_cirr_val_predictions_relative(clip_model , relative_val_dataset: CIRRDataset,
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


def cirr_val_retrieval(combining_function: callable, clip_model, preprocess: callable):
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
    parser.add_argument("--blip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--blip-model-path", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")

    args = parser.parse_args()

    if args.dataset == 'CIRR':
        blip_validate_cirr(args.blip_model_name, args.blip_model_path, args.transform, args.target_ratio)




def blip_validate_cirr(blip_model_name, blip_model_path, transform, target_ratio):

    # clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
    blip_model, _, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type="pretrain", is_eval=False, device=device)

    checkpoint = torch.load(blip_model_path, map_location=device)
    msg = blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)
    print("Missing keys {}".format(msg.missing_keys))

    input_dim = 224

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    # Define the validation datasets
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)

    val_index_features, val_index_names = extract_index_blip_features(classic_val_dataset, blip_model)
    # 
    results = compute_cirr_val_metrics(relative_val_dataset, blip_model, val_index_features,
                                        val_index_names, txt_processors)
    group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results
    results_dict = {
        'group_recall_at1': group_recall_at1,
        'group_recall_at2': group_recall_at2,
        'group_recall_at3': group_recall_at3,
        'recall_at1': recall_at1,
        'recall_at5': recall_at5,
        'recall_at10': recall_at10,
        'recall_at50': recall_at50,
        'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
        'arithmetic_mean': mean(results),
        'harmonic_mean': harmonic_mean(results),
        'geometric_mean': geometric_mean(results)
    }
    print(json.dumps(results_dict, indent=4))

if __name__ == '__main__':
    main()
