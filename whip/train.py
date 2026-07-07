from pathlib import Path

import torch
import typer
import wandb
from torch.utils.data import DataLoader

from .dataset import WhipDataset
from .model import Flow
from .types import batch_to_device

app = typer.Typer()


@app.command()
def main(
    data: Path,
    out: Path = typer.Option(Path("logs/whip")),
    name: str = typer.Option("whip"),
    sequence_length: int = typer.Option(90),
    batch: int = typer.Option(64),
    epochs: int = typer.Option(12),
    workers: int = typer.Option(16),
    lr: float = typer.Option(1e-3),
    weight_decay: float = typer.Option(1e-5),
    dim: int = typer.Option(768),
    depth: int = typer.Option(8),
    heads: int = typer.Option(12),
    log_every: int = typer.Option(50),
    compile_model: bool = typer.Option(False),
) -> None:
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out.mkdir(parents=True, exist_ok=True)

    train_set = WhipDataset(data, "train", sequence_length)
    train_loader = DataLoader(
        train_set,
        batch,
        True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 1,
    )

    model = Flow(actions=len(train_set.actions), dim=dim, depth=depth, heads=heads).to(device)
    model = torch.compile(model) if compile_model else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = (len(train_set) + batch - 1) // batch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=steps_per_epoch)
    run = wandb.init(project="whip", name=name, dir=out)
    global_step = 0

    for epoch in range(epochs):
        model.train()
        for batch_data in train_loader:
            global_step += 1
            batch_data = batch_to_device(batch_data, device)
            loss = model.loss(batch_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            if global_step % log_every == 0:
                run.log(
                    {"train/loss": loss.item(), "lr": current_lr, "epoch": epoch, "trainer/global_step": global_step}
                )

        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "global_step": global_step},
            out / "last.pt",
        )


if __name__ == "__main__":
    app()
