# Kaggle second place solution 'IceMix'

This is an overview of the different models used in the second place solution for the [Kaggle IceCube Neutrino Detection Competition](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice). The solution was developed by [DrHB](https://www.kaggle.com/drhabib) and [Iafoss](https://www.kaggle.com/iafoss).

Five different models are defined with the config files. Weights can be downloaded in [ice-cube-final-models](https://www.kaggle.com/datasets/drhabib/ice-cube-final-models).

Original repository can be found [here](https://github.com/DrHB/icecube-2nd-place).


## Loading weights into GraphNet

The state dict of the models is an exact copy of the one used in the competition, so further state_dict keys might be needed to load the models. Here is some pseudo-code which might help you load the weights into GraphNet:

```python
checkpoint = torch.load(path_to_checkpoint, torch.device("cpu"))
if "state_dict" in checkpoint:
       checkpoint = checkpoint["state_dict"]
new_checkpoint = {('_tasks.0._affine.weight' if k == 'proj_out.weight' else '_tasks.0._affine.bias' if k == 'proj_out.bias' else '_gnn.' + k): v for k, v in checkpoint.items()}
model.load_state_dict(new_checkpoint, strict=False)

```

## Direction labels

Another important thing to note is that the `prediction_x` and `prediction_y` of the direction is flipped due to a bug during the training. This means that the `prediction_x` should be used as `prediction_y` and vice versa.

Here is some pseudo-code which can be used as a label for the direction:


```python
class Direction_flipped(Label):
    """Class for producing particle direction/pointing label."""

    def __init__(
        self,
        key: str = "direction",
        azimuth_key: str = "azimuth",
        zenith_key: str = "zenith",
    ):
        """Construct `Direction`.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `graph[key] = label`.
            azimuth_key: The name of the pre-existing key in `graph` that will
                be used to access the azimiuth angle, used when calculating
                the direction.
            zenith_key: The name of the pre-existing key in `graph` that will
                be used to access the zenith angle, used when calculating the
                direction.
        """
        self._azimuth_key = azimuth_key
        self._zenith_key = zenith_key

        # Base class constructor
        super().__init__(key=key)

    def __call__(self, graph: Data) -> torch.tensor:
        """Compute label for `graph`."""
        x = torch.cos(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        ).reshape(-1, 1)
        y = torch.sin(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        ).reshape(-1, 1)
        z = torch.cos(graph[self._zenith_key]).reshape(-1, 1)
        return torch.cat((y, x, z), dim=1) # Flipped x and y
```

## Hard Local Coincidence feature

One last thing to note is regarding the **Hard Local Coincidence** (HLC) feature. The HLC feature is a binary feature that indicates whether a hit fulfills a certain criteria (see [The IceCube Neutrino Observatory: Instrumentation and Online Systems](https://arxiv.org/pdf/1612.05093.pdf), p.49).

The model is implemented in a way that it expects the `hlc` key to be `True` if the hit fulfills the criteria.

In the Kaggle dataset, this feature was stored in the `auxiliary` key. Here a `False` key means that the hit fulfills the criteria, and a `True` if not. Usually this is not the case for the `hlc` key in some other IceTray datasets, so that is something to be aware of.

Here is some pseudo-code which can be used to change the value of the `hlc` key in the [Detector](https://github.com/ArturoLlorente/graphnet/blob/train/northern_2nd_position/src/graphnet/models/detector/icecube.py) class:

```python

def _hlc(self, x: torch.tensor) -> torch.tensor:
       return torch.where(torch.eq(x, 0), torch.ones_like(x), torch.ones_like(x)*0)


```

## Stacking

The original models were stacked using a simple linear combination of the predictions. Here is some pseudo-code which can be used to stack the models:

```python
MODEL_NAMES = [
    "B_d32",
    "B_d64",
    "B_d32_4rel",
    "B+DynEdge_d64",
    "S+DynEdge_d32",
]

MODEL_WEIGHTS = {
    "B_d32": 0.08254897,
    "B_d64": 0.15350807,
    "B_d32_4rel": 0.19367443,
    "B+DynEdge_d64": 0.23597202,
    "S+DynEdge_d32": 0.3342965,
}

MODEL_PATH = {
    "B_d32": weight_path + "baselineV3_BE_globalrel_d32_0_6ema.pth",
    "B_d64": weight_path + "baselineV3_BE_globalrel_d64_0_3emaFT_2.pth",
    "B_d32_4rel": weight_path + "VFTV3_4RELFT_7.pth",
    "B+DynEdge_d64": weight_path + "V22FT6_1.pth",
    "S+DynEdge_d32": weight_path + "V23FT5_6.pth",
}


def inference(
    device: torch.device,
    test_dataloader: DataLoader,
) -> pd.DataFrame:

    models, weights = [], []
    for model_name in MODEL_NAMES:
        model = build_model(model_name)
        model.eval()
        model.inference()
        checkpoint_path = MODEL_PATH[model_name]
        checkpoint = torch.load(checkpoint_path, torch.device("cpu"))
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        new_checkpoint = {
            (
                "_tasks.0._affine.weight"
                if k == "proj_out.weight"
                else "_tasks.0._affine.bias"
                if k == "proj_out.bias"
                else "_gnn." + k
            ): v
            for k, v in checkpoint.items()
        }

        model.load_state_dict(new_checkpoint)
        model.to(device)
        models.append(model)
        weights.append(MODEL_WEIGHTS[model_name])

    weights = torch.FloatTensor(weights)
    weights /= weights.sum()

    event_nos, zenith, azimuth, preds = [], [], [], []
    with torch.no_grad():
        for batch in test_dataloader:
            pred = (
                torch.stack(
                    [
                        torch.nan_to_num(model(batch.to(device))[0]).clip(
                            -1000, 1000
                        )
                        for model in models
                    ],
                    -1,
                ).cpu()
                * weights
            ).sum(-1)
            preds.append(pred)
            event_nos.append(batch.event_no)
            zenith.append(batch.zenith)
            azimuth.append(batch.azimuth)
    preds = torch.cat(preds).to("cpu").detach().numpy()
    columns = [
        "direction_x",
        "direction_y",
        "direction_z",
        "direction_kappa",
    ]

    results = pd.DataFrame(preds, columns=columns)
    results["event_no"] = torch.cat(event_nos).to("cpu").detach().numpy()

    results["zenith"] = torch.cat(zenith).to("cpu").numpy()
    results["azimuth"] = torch.cat(azimuth).to("cpu").numpy()

    return results

```
