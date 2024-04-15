import json
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import CIRRDataset, targetpad_transform, base_path
from utils import device, extract_index_blip_features
from lavis.models import load_model_and_preprocess
import argparse

def generate_cirr_test_submissions(file_name: str, blip_model, preprocess, txt_processors, rerank):
    """
   Generate and save CIRR test submission files to be submitted to evaluation server
   :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                           features
   :param file_name: file_name of the submission
   :param clip_model: CLIP model
   :param preprocess: preprocess pipeline
   """
    # Define the dataset and extract index features
    classic_test_dataset = CIRRDataset('test1', 'classic', preprocess)
    index_features, index_names = extract_index_blip_features(classic_test_dataset, blip_model)
    relative_test_dataset = CIRRDataset('test1', 'relative', preprocess)

    # Generate test prediction dicts for CIRR
    pairid_to_predictions, pairid_to_group_predictions = generate_cirr_test_dicts(relative_test_dataset, blip_model,
                                                                                  index_features, index_names, txt_processors, rerank)

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_predictions)
    group_submission.update(pairid_to_group_predictions)

    # Define submission path
    submissions_folder_path = base_path / "submission" / 'CIRR'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    if rerank:
        file_name = file_name + f'_{rerank}'

    print(f"Saving CIRR test predictions")
    with open(submissions_folder_path / f"recall_submission_{file_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"recall_subset_submission_{file_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)


def generate_cirr_test_dicts(relative_test_dataset: CIRRDataset, blip_model, index_features: torch.tensor,
                             index_names: List[str], txt_processors, rerank) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Compute test prediction dicts for CIRR dataset
    :param relative_test_dataset: CIRR test dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: test index features
    :param index_names: test index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: Top50 global and Top3 subset prediction for each query (reference_name, caption)
    """

    # Generate predictions
    predicted_sim, reference_names, group_members, pairs_id, captions_all, name2feat = \
        generate_cirr_test_predictions(blip_model, relative_test_dataset, index_names,
                                       index_features, txt_processors)

    print(f"Compute CIRR prediction dicts")
    # Compute the distances and sort the results
    distances = 1 - predicted_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]


    # re-rank
    if rerank:
        print('reranking now')
        i = 0
        step = 50
        top = 50
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


    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names),
                                                                                             -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_predictions = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                             zip(pairs_id, sorted_index_names)}
    pairid_to_group_predictions = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                   zip(pairs_id, sorted_group_names)}

    return pairid_to_predictions, pairid_to_group_predictions


def generate_cirr_test_predictions(blip_model, relative_test_dataset: CIRRDataset, index_names: List[str], 
                                   index_features: torch.tensor, txt_processors) -> \
        Tuple[torch.tensor, List[str], List[List[str]], List[str]]:
    """
    Compute CIRR predictions on the test set
    :param clip_model: CLIP model
    :param relative_test_dataset: CIRR test dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                           features
    :param index_features: test index features
    :param index_names: test index names

    :return: predicted_features, reference_names, group_members and pairs_id
    """
    print(f"Compute CIRR test predictions")

    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32,
                                      num_workers=4, pin_memory=True)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features[1]))

    # Initialize pairs_id, predicted_features, group_members and reference_names
    pairs_id = []
    group_members = []
    reference_names = []
    distance = []
    captions_all = []

    for batch_pairs_id, batch_reference_names, captions, batch_group_members in tqdm(
            relative_test_loader):  # Load data
        
        batch_group_members = np.array(batch_group_members).T.tolist()
        captions = [txt_processors["eval"](caption) for caption in captions]

        # Compute the predicted features
        with torch.no_grad():
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if len(captions) == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_distance = blip_model.inference(reference_image_features, index_features[0], captions)
            distance.append(batch_distance)
            captions_all += captions


        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)
        pairs_id.extend(batch_pairs_id)

    distance = torch.vstack(distance)

    return distance, reference_names, group_members, pairs_id, captions_all, name_to_feat


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = ArgumentParser()
    parser.add_argument("--blip-model-name", default="blip2_cir_cat", type=str)
    parser.add_argument("--model-path", type=str, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--backbone", type=str, default="pretrain", help="pretrain for vit-g, pretrain_vitL for vit-l")
    parser.add_argument("--rerank", type=str2bool, default=False)
    args = parser.parse_args()
    # blip model
    blip_model, _, txt_processors = load_model_and_preprocess(name=args.blip_model_name, model_type=args.backbone, is_eval=False, device=device)
    
    checkpoint_path = args.model_path

    checkpoint = torch.load(checkpoint_path, map_location=device)
    msg = blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)
    print("Missing keys {}".format(msg.missing_keys))

    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)

    generate_cirr_test_submissions(f'{args.blip_model_name}_2', blip_model, preprocess, txt_processors, args.rerank)


if __name__ == '__main__':
    main()
