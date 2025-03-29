import numpy as np
import os
import argparse
from typing import List
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./cache")
    return parser.parse_args()

def download_cath_dataset(download_dir: str) -> None:
    """
    Download the CATH dataset from the GitHub repository to the download directory.

    Args:
        download_dir: The directory to download the CATH dataset to.
    """
    cath_repo_url = f"https://github.com/wouterboomsma/cath_datasets.git"
    repo_dir = os.path.join(download_dir, "cath_datasets")
    if not os.path.exists(repo_dir):
        os.system(f"git clone {cath_repo_url} {repo_dir}")
    else:
        print(f"CATH dataset already exists in {repo_dir}")

def get_label_to_index_mapping(labels: List[List[int]]) -> dict:
    """
    Get the label to index mapping from the preprocessed data.
    """
    label_as_str = [label2str(label) for label in labels]
    label_counter = Counter(label_as_str)
    # sort the labels by the name
    sorted_labels = sorted(label_counter.keys())
    label_to_index = {label: index for index, label in enumerate(sorted_labels)}
    return label_to_index

def label2str(label: List[int]) -> str:
    """
    Convert the label to a string.
    """
    return ".".join([str(l) for l in label])


def preprocess_data(npz_file: str, dataset_name: str, output_dir: str = "processed_data") -> None:
    """
    Preprocess the data from the CATH dataset.

    Args:
        npz_file: The path to the preprocessed data.
        dataset_name: The name of the dataset.
        output_dir: The directory to save the preprocessed data to.
    """
    # load data
    data = np.load(npz_file, allow_pickle=True)
    positions, labels, n_atoms, atom_types = data['positions'], data['labels'], data['n_atoms'], data['atom_types']
    num_samples, length, ndim = positions.shape
    label_to_idx = get_label_to_index_mapping(labels)

    # preprocess data
    split_indices = data['split_start_indices']
    processed_positions, processed_labels = [], []
    
    for i in range(num_samples):
        protein_positions = positions[i]
        effective_protein_positions = positions[i][:n_atoms[i]]
        protein_atom_types = atom_types[i][:n_atoms[i]]
        ca_positions = effective_protein_positions[np.array(protein_atom_types) == b'CA']

        if len(ca_positions) != len(effective_protein_positions):
            raise f"abnormal data observed indexed with {i}"

        # normalize
        protein_positions -= ca_positions.mean(axis=0)

        # print(padded_positions.shape)
        processed_positions.append(protein_positions)
        label_idx = label_to_idx[label2str(labels[i])]
        processed_labels.append(label_idx)

    # Convert to numpy arrays
    processed_positions = np.array(processed_positions, dtype=object)  # Variable-length lists
    processed_labels = np.array(processed_labels)

    # raise "Stop"
    # Save preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(f"{output_dir}/{dataset_name}.npz", positions=processed_positions, labels=processed_labels,
                        split_indices=split_indices, label_mapping=label_to_idx)
    print(f"Preprocessed data saved to {output_dir}/{dataset_name}.npz")


def create_cross_validation_splits(npz_file: str, dataset_name: str, output_dir: str = "cv_data") -> None:
    """
    Create cross-validation splits for the preprocessed data.

    Args:
        npz_file: The path to the preprocessed data.
        dataset_name: The name of the dataset.
        output_dir: The directory to save the cross-validation splits to.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load the preprocessed data
    data = np.load(npz_file, allow_pickle=True)
    positions, labels, split_indices, label_mapping = data['positions'], data['labels'], data['split_indices'], data['label_mapping']



    # Generate 10 folds
    print(f"Creating cross-validation splits for {dataset_name}")
    for fold in range(10):
        test_start, test_end = split_indices[fold], split_indices[fold + 1] if fold < 9 else len(positions)
        test_indices = np.arange(test_start, test_end)

        train_indices = np.concatenate([
            np.arange(split_indices[i], split_indices[i + 1]) for i in range(9) if i != fold  # Exclude fold 9
        ])
        if fold != 9:  # For the last fold, handle last section differently
            train_indices = np.concatenate([train_indices, np.arange(split_indices[9], len(positions))])

        train_positions = positions[train_indices]
        train_labels = labels[train_indices]
        test_positions = positions[test_indices]
        test_labels = labels[test_indices]

        np.savez_compressed(f"{output_dir}/{dataset_name}_train_fold{fold}.npz",
                            positions=train_positions, labels=train_labels, label_mapping=label_mapping)
        np.savez_compressed(f"{output_dir}/{dataset_name}_test_fold{fold}.npz",
                            positions=test_positions, labels=test_labels, label_mapping=label_mapping)

        print(f"Fold {fold}: Train={len(train_positions)}, Test={len(test_positions)}")


def main():
    args = parse_args()
    download_cath_dataset(args.save_dir)

    # preprocess data
    topo_src_path = os.path.join(args.save_dir, "cath_datasets", "cath_20topo_ca.npz")
    arch_src_path = os.path.join(args.save_dir, "cath_datasets", "cath_10arch_ca.npz")
    processed_dir = os.path.join(args.save_dir, "processed_data")
    # preprocess_data(topo_src_path, "cath_20topo_ca", processed_dir)
    # preprocess_data(arch_src_path, "cath_10arch_ca", processed_dir)

    # create cross-validation splits
    cv_path = os.path.join(args.save_dir, "cv_data")
    create_cross_validation_splits(os.path.join(processed_dir, "cath_20topo_ca.npz"), "cath_20topo_ca", cv_path)
    create_cross_validation_splits(os.path.join(processed_dir, "cath_10arch_ca.npz"), "cath_10arch_ca", cv_path)

if __name__ == "__main__":
    main()
    