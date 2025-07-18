import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ptflops import get_model_complexity_info
from pytorch_lightning import seed_everything

from beat_this.dataset import BeatDataModule
from beat_this.inference import load_checkpoint
from beat_this.model.pl_module import PLBeatThis


seed_everything(0, workers=True)


def datamodule_setup(checkpoint, num_workers, datasplit):
    print("Creating datamodule...")
    data_dir = Path(__file__).parent.parent / "data"
    hparams = checkpoint["datamodule_hyper_parameters"]
    hparams["predict_datasplit"] = datasplit
    hparams["data_dir"] = data_dir
    hparams["logits_dir"] = data_dir / "output_npy_files"
    hparams["num_workers"] = num_workers
    datamodule = BeatDataModule(**hparams)
    datamodule.setup(stage="fit")  
    datamodule.setup(stage="predict")
    return datamodule


def plmodel_setup(checkpoint):
    model = PLBeatThis(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model


def compute_flops(model, input_shape):
    model.eval()
    with torch.no_grad():
        macs, params = get_model_complexity_info(
            model,
            input_res=input_shape,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
    return macs, params


def main(args):
    checkpoint = load_checkpoint(args.model_ckpt)
    model_type = checkpoint["hyper_parameters"].get("model_type", "Unknown")

    datamodule = datamodule_setup(checkpoint, args.num_workers, args.datasplit)
    model = plmodel_setup(checkpoint).model 

 
    sample_batch = next(iter(datamodule.predict_dataloader()))

   
    print("Sample batch type:", type(sample_batch))
    print("Sample batch keys:", list(sample_batch.keys()))

    if isinstance(sample_batch, dict) and "spect" in sample_batch:
        input_tensor = sample_batch["spect"]
    else:
        raise ValueError("Unexpected batch format: cannot extract spectrogram input.")

    input_tensor = input_tensor[0:1]  # Take one sample
    print(input_tensor.shape)
    input_shape = tuple(input_tensor.shape[1:])  
    print(input_shape)
    macs, params = compute_flops(model, input_shape)

    print(f"\n FLOPs/Params for model type: {model_type}")
    print(f"Input shape: {input_shape}")
    print(f"FLOPs (MACs): {macs}")
    print(f"Parameters: {params}")

  
    if args.save_results:
        df = pd.DataFrame([{
            "model_type": model_type,
            "input_shape": input_shape,
            "FLOPs_MACs": macs,
            "Params": params
        }])
        df.to_csv(args.save_results, index=False)
        print(f" Saved FLOPs report to: {args.save_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FLOPs/Params using real data")
    parser.add_argument("--model-ckpt", type=str, required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--datasplit", type=str, choices=[ "val", "test"], default="val")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-results", type=str, default=None, help="Path to CSV to save the results")
    args = parser.parse_args()

    main(args)
