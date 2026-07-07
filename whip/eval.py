from pathlib import Path

import torch
import typer
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader

from . import metrics as metric_lib
from .dataset import WhipDataset
from .model import Flow
from .types import Batch, batch_to_device

app = typer.Typer()


@app.command()
def main(
    data: Path,
    checkpoint: Path,
    split: str = typer.Option("test"),
    combo: str = typer.Option(...),
    samples: int = typer.Option(4),
    batch: int = 64,
    workers: int = 16,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WhipDataset(data, split, stride=45)
    loader = DataLoader(dataset, batch, False, num_workers=workers, pin_memory=True)
    model = Flow(actions=len(dataset.actions)).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    for key, value in evaluate(model, loader, device, combo, samples).items():
        print(f"test/{split}/{combo}/metric/{key}: {value:.4f}")


@torch.no_grad()
def evaluate(model: Flow, loader: DataLoader, device: torch.device, combo: str, samples: int = 4) -> dict[str, float]:
    model.eval()
    preds = []
    gt = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        gt.append(batch["joints"].cpu())
        preds.append(predict(model, batch, combo, samples).cpu())
    return metric_lib.metrics(torch.cat(preds), torch.cat(gt))


@torch.no_grad()
def predict(model: Flow, batch: Batch, combo: str, samples: int) -> Float[Tensor, "B T J 3"]:
    return torch.stack([model.sample(batch, combo=combo) for _ in range(samples)]).mean(0)


if __name__ == "__main__":
    app()
