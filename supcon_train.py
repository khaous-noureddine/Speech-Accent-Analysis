import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from supcon_data import SupConSpeechDataset, SupConBatchSampler, collate_supcon
from supcon_xlsr import SupConXLSR

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
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/wav2vec2-large-xlsr-53",
        help="Pretrained model name or path (default: facebook/wav2vec2-large-xlsr-53)"
    )
    parser.add_argument(
        "--proj_hidden_dim",
        type=int,
        default=512,
        help="Dimension of the projection head output (default: 512)"
    )
    parser.add_argument(
        "--proj_out_dim",
        type=int,
        default=256,
        help="Dimension of the projection head output (default: 256)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for contrastive loss (default: 0.1)"
    )
    parser.add_argument(
        "--ctc_lambda",
        type=float,
        default=0.1,
        help="Weight for CTC loss (default: 0.1)"
    )
    parser.add_argument(
        "--min_frozen_layer",
        type=int,
        default=18,
        help="Minimum layer to freeze during training (default: 18)"
    )
    parser.add_argument(
        "--max_frozen_layer",
        type=int,
        default=24,
        help="Maximum layer to freeze during training (default: 24)"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,   
        default=32,
        help="Vocabulary size for CTC loss (default: 32)"
    )

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

    # batch = next(iter(train_loader))
    # print("Batch keys:", batch.keys())
    # print("Audio shape:", batch["audio"][0].shape)
    # print("Labels shape:", batch["labels"].shape)


    model = SupConXLSR(
        model_name=args.model_name,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_out_dim=args.proj_out_dim,
        vocab_size=args.vocab_size,
        ctc_lambda=args.ctc_lambda,
        temperature=args.temperature,
        min_frozen_layer=args.min_frozen_layer,
        max_frozen_layer=args.max_frozen_layer,
    )
    

if __name__ == "__main__":
    main() 