from argparse import ArgumentParser
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from dataset.mvtec import MVtecDataset
from lightning_models import FewShotADLit

categories = [
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def parse_args():
    parser = ArgumentParser(description="Inference Few-shot Anomaly Detection model")

    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument('--model_name', type=str, default=None, required=False, help="Feature encoder name")
    parser.add_argument('--batch_size', type=int, default=1, help="Input batch size")
    parser.add_argument('--max_samples', type=int, default=None, required=False, help="data subset size")
    parser.add_argument('--save_dir', type=str, default='./fs_results', help="Output directory")

    args = parser.parse_args()
    print(args)

    return args


def main():
    args = parse_args()

    for category in categories:
        data_path_category = Path(args.data_dir) / category

        print("=" * 50)
        print(f"Inference on: {category}")
        print("=" * 50)

        dataset = MVtecDataset(
            root_dir=data_path_category,
            mode="test",
            few_shot_k=1
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size, 
            shuffle=False
        )

        model = FewShotADLit(args)
        trainer = L.Trainer()

        preds = trainer.predict(model, dataloader)


if __name__ == "__main__":
    main()