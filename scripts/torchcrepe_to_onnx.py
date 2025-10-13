from pathlib import Path

import torchcrepe
import torch
import click

@click.command()
@click.argument('model', type=click.Choice(['tiny', 'full']))
@click.argument('output_dir', type=click.Path(exists=True, dir_okay=True, path_type=Path), default=Path('.'))
def export(model, output_dir):
    torchcrepe.load.model('cpu', model)

    torch.onnx.export(
        torchcrepe.infer.model,
        (torch.sin(torch.arange(1024) * 2 * torch.pi * 440 / 8000)[None, :],),
        output_dir / f"{model}.onnx",
        input_names=["input"],
        dynamo=True,
    )

if __name__ == '__main__':
    export()

