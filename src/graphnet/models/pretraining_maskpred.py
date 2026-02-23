"""Self-supervised pretraining using BERT-style mask prediction."""

from typing import Any, Tuple, Union, List, Type, Optional, Dict
import os

import torch
from torch.optim.adam import Adam
from torch.functional import Tensor

from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.utils import mask_select
from torch_geometric.data import Data

from torch_scatter import scatter

from graphnet.models import Model
from graphnet.models.easy_model import EasySyntax
from graphnet.models.task.task import UnsupervisedTask

from graphnet.training.loss_functions import MSELoss
from graphnet.training.loss_functions import NegCosLoss


class standard_maskpred_net(Model):
    """A small NN that is used as a default."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 1000,
        out_dim: int = 5,
        nb_linear: int = 5,
    ) -> None:
        """Construct the default NN."""
        super().__init__()

        self.activation = torch.nn.SELU()

        self.lin_net = torch.nn.ModuleList()
        for i in range(nb_linear):
            if i == 0:
                self.lin_net.append(torch.nn.Linear(in_dim, hidden_dim))
            else:
                self.lin_net.append(torch.nn.Linear(hidden_dim, hidden_dim))

        self.final_proj = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, data: Union[Data, Tensor]) -> Tensor:
        """Forward pass,linear layers plus final projection."""
        if isinstance(data, Data):
            x_hat = data.x
        else:
            x_hat = data
        x_hat = self.lin_net[0](x_hat)
        x_hat = self.activation(x_hat)
        for i in range(1, len(self.lin_net)):
            x_hat = x_hat + self.lin_net[i](x_hat)
            x_hat = self.activation(x_hat)

        x_hat = self.final_proj(x_hat)

        return x_hat


class default_mask_augment(Model):
    """A module that produces masked nodes, target, mask and charge summary."""

    def __init__(
        self,
        masked_ratio: float = 0.25,
        masked_feat: List[int] = [0, 1, 2],
        learned_masking_value: bool = True,
        hlc_pos: Optional[int] = None,
    ) -> None:
        """Construct the augmentation."""
        super().__init__()
        self.ratio = masked_ratio
        self.hlc_pos = hlc_pos
        self.masked_feat = masked_feat
        self.learned_value = learned_masking_value

        if self.learned_value:
            print(
                """warning: can currently only mask adjacent features,
                  e.g. only (x,y,z) or only (t,q) but not e.g. (x,t,q)"""
            )
            self.values = torch.nn.Parameter(
                torch.randn(1, len(self.masked_feat))
            )

    def forward(self, data: Data) -> Tuple[Any, Any, Any, Any]:
        """Forward pass."""
        auged = data.clone()

        charge_tensor = torch.pow(10, data.x[:, -1]).view(-1, 1)
        charge_sums = torch.log10(
            scatter(
                src=charge_tensor,
                index=data.batch,
                dim=0,
                reduce="sum",
            )
        )

        rand_score = torch.rand_like(data.batch.to(dtype=torch.bfloat16))
        if self.hlc_pos is not None:
            rand_score = rand_score + auged.x[:, self.hlc_pos].view(1, -1)
            rand_score = rand_score.view(-1)

        ind = topk(x=rand_score, ratio=self.ratio, batch=data.batch)

        mask = torch.ones_like(data.batch.to(dtype=torch.bfloat16))
        mask[ind] = 0

        target = mask_select(src=auged.x, dim=0, mask=~mask.bool())[
            :, self.masked_feat
        ]
        if not self.learned_value:
            auged.x[:, self.masked_feat] = auged.x[
                :, self.masked_feat
            ] * mask.view(-1, 1)
        else:
            auged.x[ind, self.masked_feat[0] : self.masked_feat[-1] + 1] = (
                self.values
            )

        # returned mask is zero at the target position and 1 else
        return auged, target, mask, charge_sums


class default_loss_calc(Model):
    """Applies the default loss logic that matches the default augment."""

    def __init__(
        self,
        encoder_out_dim: int = 5,
        masked_feat: List[int] = [0, 1, 2],
        mask_pred_net: Optional[Model] = None,
        final_loss: str = "mse",
        default_hidden_dim: int = 1000,
        default_nb_linear: int = 5,
    ) -> None:
        """Construct the loss calc."""
        super().__init__()
        if mask_pred_net is None:
            print(
                "no custom net for mask prediction specified; using a standard net"
            )
            self.rep = standard_maskpred_net(
                in_dim=encoder_out_dim,
                hidden_dim=default_hidden_dim,
                out_dim=len(masked_feat),
                nb_linear=default_nb_linear,
            )
        else:
            assert mask_pred_net.nb_outputs == len(
                masked_feat
            ), f'make sure that your "mask_pred_net" has number of output feats equal to nb of masked feats ({len(masked_feat)})'
            self.rep = mask_pred_net

        self.charge_net = torch.nn.Linear(encoder_out_dim, 1)

        assert final_loss in [
            "cosine",
            "mse",
        ], "can only choose from [cosine, mse] for loss function"
        if final_loss == "cosine":
            self.loss_func = NegCosLoss()
        elif final_loss == "mse":
            self.loss_func = MSELoss()

    def forward(
        self,
        pred: Tuple[Tensor, Tensor],
        data: Data,
        aux: Tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        """Forward pass."""
        lat, cls_tensor = pred
        target, mask, charge_sums = aux
        rep = self.rep(lat)  # type: ignore

        nodes = rep[~mask.bool()]
        btch = data.batch[~mask.bool()]

        loss = scatter(
            src=self.loss_func(nodes, target, return_elements=True),
            index=btch,
            reduce="mean",
            dim=0,
        ).view(-1, 1)

        pred_charge = self.charge_net(cls_tensor)
        loss = loss + (charge_sums - pred_charge) ** 2

        return loss


class mask_pred_frame(EasySyntax):
    """The BERT-Style mask prediction module.

    Should be compatible with any module as long as it does not change
    the length of the input data in dense rep.

    One needs to provide the encoder, i.e. the model to be pretrained,
    and an UnsupervisedTask, which is constructed from an
    augmentation_like module and a loss calculation (see the defaults
    above).
    """

    def __init__(
        self,
        encoder: Model,
        bert_task: UnsupervisedTask,
        encoder_out_dim: Optional[int] = None,
        need_charge_rep: bool = False,
        optimizer_class: Type[torch.optim.Optimizer] = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
    ) -> None:
        """Construct the pretraining framework."""
        super().__init__(
            tasks=bert_task,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_config=scheduler_config,
        )
        self.aug_task = bert_task
        self.backbone = encoder

        if encoder_out_dim is None:
            assert (
                encoder.nb_outputs > 0
            ), 'make sure to either specify "encoder_out_dim" or have a ".nb_outputs" in your encoder'
            lat_dim = encoder.nb_outputs
        else:
            lat_dim = encoder_out_dim

        self.need_charge_rep = need_charge_rep
        if need_charge_rep:
            self.lin_layer_scatter = torch.nn.Linear(lat_dim, lat_dim)

    def forward(self, data: Union[Data, List[Data]]) -> List[Tensor]:
        """Forward pass, produce latent view compare against target.

        per default predict summary value.
        """
        if not isinstance(data, Data):
            data = data[0]

        # first part of the task: augmentation
        aug = self.aug_task.augment(data)

        data_hat = self.backbone(aug)
        if self.need_charge_rep:
            reduce_list = ["sum", "mean", "max", "min"]
            cls_tensor = scatter(
                src=data_hat, index=data.batch, reduce=reduce_list[0], dim=0
            )
            for i in range(1, len(reduce_list)):
                cls_tensor = cls_tensor + scatter(
                    src=data_hat,
                    index=data.batch,
                    reduce=reduce_list[i],
                    dim=0,
                )
            cls_tensor = self.lin_layer_scatter(cls_tensor)
            data_hat = data_hat, cls_tensor

        assert (
            len(data_hat[0].shape) == 2
        ), "dense data representation [n_pulses, lat_dim] is required for the processed tensor as an artifact"

        # second part of the task: loss calc
        loss = self.aug_task.compute_loss(data_hat, data)

        # loss is returned as a list to comply with the graphnet predict functionality
        return [loss]

    def validate_tasks(self) -> None:
        """Verify that self._tasks contain compatible elements."""
        accepted_tasks = UnsupervisedTask
        for task in self._tasks:
            assert isinstance(task, accepted_tasks)

    def shared_step(self, batch: List[Data], batch_idx: int) -> Tensor:
        """Perform shared step.

        Applies the forward pass and the following loss calculation,
        shared between the training and validation step.
        """
        loss = self(batch)
        if isinstance(loss, list):
            assert len(loss) == 1
            loss = loss[0]
        return torch.mean(loss, dim=0)

    def give_encoder_model(self) -> Model:
        """Return the pretrained encoder model."""
        return self.backbone

    def save_pretrained_model(self, save_path: str) -> None:
        """Automates the saving of the pretrained encoder."""
        model = self.backbone

        run_name = "pretrained_model"

        save_path = os.path.join(save_path, run_name)
        print("saving to", save_path)
        os.makedirs(save_path, exist_ok=True)

        model.save_state_dict(f"{save_path}/state_dict.pth")
        model.save_config(f"{save_path}/model_config.yml")
