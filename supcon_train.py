import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from supcon_data import SupConSpeechDataset, SupConBatchSampler, collate_supcon

def main():
    parser = argparse.ArgumentParser(
        description="Supervised Contrastive Training of XLSR model"
    )

    # Data:
    parser.add_argument(
        "--arctic_parquet_path",
        type=Path,
        required=True,
        help="Path to ARCTIC parquet file"
    )

    parser.add_argument(
        "--l2_arctic_parquet_path",
        type=Path,
        required=True,
        help="Path to L2-ARCTIC parquet file"
    )
    parser.add_argument(
        "--k_utterances",
        type=int,
        default=20,
        help="Number of utterances per speaker in each batch"
    )
    parser.add_argument(
        "--s_speakers",
        type=int,
        default=25,
        help="Number of speakers per utterance in each batch"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=1000,
        help="Number of batches per epoch"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Sample rate for audio processing (default: 16000)"
    )
    parser.add_argument(
        "--max_audio_len_s",
        type=float,
        default=10.0,
        help="Maximum audio length in seconds (default: 10.0s)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of worker processes for data loading (default: 4)"
    )


    # Model:




    # Training:
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to use for training (default: cuda)"
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("./outputs"),
        help="Directory to save model checkpoints and logs (default: ./outputs)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Active le mode verbeux"
    )

    args = parser.parse_args()


    train_dataset = SupConSpeechDataset(
        parquet_paths={
            "arctic": args.arctic_parquet_path,
            "l2_arctic": args.l2_arctic_parquet_path,
        },
        split=args.split,
        sample_rate=args.sample_rate,
        max_audio_len_s=args.max_audio_len_s,
    )

    sampler = SupConBatchSampler(
        train_dataset,
        k_utterances=args.k_utterances,
        s_speakers=args.s_speakers,
        n_batches=args.n_batches,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collate_supcon,
        num_workers=args.num_workers,
    )

    batch = next(iter(train_loader))
    print("Batch keys:", batch.keys())
    print("Audio shape:", batch["audio"].shape)
    print("Labels shape:", batch["labels"].shape)


if __name__ == "__main__":
    main()