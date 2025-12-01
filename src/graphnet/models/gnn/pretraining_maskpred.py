from typing import Tuple, Union, List, Type, Optional, Dict
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
from graphnet.models.task import IdentityTask

from graphnet.training.loss_functions import MSELoss


def dense_mse_loss(reco, orig, bv):
    squared_errs = (reco - orig)**2
    losses = torch.mean(scatter(src=squared_errs, index=bv, reduce='mean', dim=0), dim=1)

    return losses.view(-1,1)

def neg_cosine_loss(reco, orig, bv):
    reco_norm = torch.nn.functional.normalize(reco, dim=1)
    orig_norm = torch.nn.functional.normalize(orig, dim=1)
    cos = -(reco_norm*orig_norm).sum(dim=1)
    losses = scatter(src=cos, index=bv, reduce='mean', dim=0)

    return losses.view(-1,1)

class standard_maskpred_net(Model):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 1000,
                 out_dim: int = 5,
                 nb_linear: int = 5,
                 ) -> None:
        super().__init__()

        self.activation = torch.nn.SELU()
        
        self.lin_net = torch.nn.ModuleList()
        for i in range(nb_linear):
            if i == 0:
                self.lin_net.append(torch.nn.Linear(in_dim,hidden_dim))
            else:
                self.lin_net.append(torch.nn.Linear(hidden_dim,hidden_dim))

        self.final_proj = torch.nn.Linear(hidden_dim, out_dim)
        

    def forward(self, data:Union[Data, Tensor]) -> Tensor:
        if isinstance(data, Data):
            x_hat = data.x
        else:
            x_hat = data
        x_hat = self.lin_net[0](x_hat)
        x_hat = self.activation(x_hat)
        for i in range(1,len(self.lin_net)):
            x_hat = x_hat + self.lin_net[i](x_hat)
            x_hat = self.activation(x_hat)

        x_hat = self.final_proj(x_hat)
        
        return x_hat

class mask_pred_augment(Model):
    def __init__(self, 
                 masked_ratio: float = 0.25,
                 masked_feat: List[int] = [0,1,2,3,4],
                 learned_masking_value: bool = True,
                 hlc_pos: int = None,
                 ) -> None:
        
        super().__init__()
        self.ratio = masked_ratio
        self.hlc_pos = hlc_pos
        self.masked_feat = masked_feat
        self.learned_value = learned_masking_value

        if self.learned_value:
            print('warning: can currently only mask adjacent features, e.g. only (x,y,z) or only (t,q) but not e.g. (x,t,q)')
            self.values = torch.nn.Parameter(torch.randn(1,len(self.masked_feat)))

    def forward(self, data: Data) -> Tuple[Union[Data,Tensor]]:
        auged = data.clone()

        rand_score = torch.rand_like(data.batch.to(dtype=torch.bfloat16))
        if self.hlc_pos is not None:
            rand_score = rand_score + auged.x[:,self.hlc_pos].view(1,-1)
            rand_score = rand_score.view(-1)

        ind = topk(x=rand_score, ratio=self.ratio, batch=data.batch)

        mask = torch.ones_like(data.batch.to(dtype=torch.bfloat16))
        mask[ind] = 0

        target = mask_select(src=auged.x, dim=0, mask=~mask.bool())[:,self.masked_feat]
        if not self.learned_value:
            auged.x[:,self.masked_feat] = auged.x[:,self.masked_feat]*mask.view(-1,1)
        else:
            auged.x[ind,self.masked_feat[0]:self.masked_feat[-1]+1] = self.values

        #returned mask is zero at the target position and 1 else
        return auged, target, mask
    
class mask_pred_frame(EasySyntax):
    def __init__(self,
                 encoder: Model,
                 encoder_out_dim: int = None,
                 masked_ratio: float = 0.25,
                 masked_feat: List[int] = [0,1,2,3,4],
                 learned_masking_value: bool = True,
                 hlc_pos: int = None,
                 mask_pred_net: Model = None,
                 default_hidden_dim: int = 1000, 
                 default_nb_linear: int = 5,
                 final_loss: str = 'mse',
                 add_charge_pred: bool = False,
                 need_charge_rep: bool = False,
                 custom_charge_target: Model = None,
                 optimizer_class: Type[torch.optim.Optimizer] = Adam,
                 optimizer_kwargs: Optional[Dict] = None,
                 scheduler_class: Optional[type] = None,
                 scheduler_kwargs: Optional[Dict] = None,
                 scheduler_config: Optional[Dict] = None,) -> None:
        
        #just because I need to specify a task
        task = IdentityTask(nb_outputs=1, target_labels=['skip'], hidden_size=2, loss_function=MSELoss())

        super().__init__(
            tasks=task,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_config=scheduler_config,
        )
        self.backbone = encoder

        self.ratio = masked_ratio

        self.augment = mask_pred_augment(masked_ratio=masked_ratio,
                                         masked_feat=masked_feat,
                                         learned_masking_value=learned_masking_value,
                                         hlc_pos=hlc_pos
                                         )

        if encoder_out_dim is None:
            assert encoder.nb_outputs > 0, 'make sure to either specify \"encoder_out_dim\" or have a \".nb_outputs\" in your encoder'
            lat_dim = encoder.nb_outputs
        else:
            lat_dim = encoder_out_dim

        if mask_pred_net is None:
            print('no custom net for mask prediction specified; using a standard net')
            self.rep = standard_maskpred_net(in_dim=lat_dim,
                                             hidden_dim=default_hidden_dim,
                                             out_dim=len(masked_feat),
                                             nb_linear=default_nb_linear)
        else:
            assert mask_pred_net.nb_outputs == len(masked_feat), f'make sure that your \"mask_pred_net\" has number of output feats equal to nb of masked feats ({len(masked_feat)})'
            self.rep = mask_pred_net
        
        assert final_loss in ['cosine', 'mse'], 'can only choose from [cosine, mse] for loss function'
        if final_loss == 'cosine':
            self.loss_func = neg_cosine_loss
        elif final_loss == 'mse':
            self.loss_func = dense_mse_loss

        self.add_charge_pred = add_charge_pred
        self.need_charge_rep = need_charge_rep
        if need_charge_rep:
            self.add_charge_pred = True
        if self.add_charge_pred:
            self.charge_net = torch.nn.Linear(lat_dim, 1)
            if need_charge_rep:
                self.lin_layer_scatter = torch.nn.Linear(lat_dim, lat_dim)
            self.custom_charge_target = custom_charge_target

            
    def forward(self, data: Union[Data, List[Data]]) -> List[Tensor]:
        if not isinstance(data, Data):
            data = data[0]

        aug, target, mask = self.augment(data)

        if not self.need_charge_rep:
            data_hat, cls_tensor = self.backbone(aug)
        else:
            data_hat = self.backbone(aug)
            reduce_list = ['sum', 'mean', 'max', 'min']
            cls_tensor = scatter(src=data_hat, index=data.batch, reduce=reduce_list[0], dim=0)
            for i in range(1,len(reduce_list)):
                cls_tensor = cls_tensor + scatter(src=data_hat, index=data.batch, reduce=reduce_list[i], dim=0)
            cls_tensor = self.lin_layer_scatter(cls_tensor)

        assert len(data_hat.shape) == 2, 'dense data representation [n_pulses, lat_dim] is required for the processed tensor as an artifact'

        rep = self.rep(data_hat)

        nodes = rep[~mask.bool()]
        btch = data.batch[~mask.bool()]

        loss = self.loss_func(reco=nodes, orig=target, bv=btch)

        if self.add_charge_pred:
            if self.custom_charge_target is None:
                charge_tensor = torch.pow(10, data.x[:,4]).view(-1,1)
                charge_sums = torch.log10(scatter(src=charge_tensor, index=data.batch, dim = 0, reduce='sum'))
            else:
                charge_sums = self.custom_charge_target(data)
            pred_charge = self.charge_net(cls_tensor)
            loss = loss + (charge_sums - pred_charge)**2

        #loss is returned as a list to comply with the graphnet predict functionality
        return [loss]

    def validate_tasks(self) -> None:
        accepted_tasks = IdentityTask
        for task in self._tasks:
            assert isinstance(task, accepted_tasks)

    def shared_step(self, batch: List[Data], batch_idx: int) -> Tensor:
        loss = self(batch)
        if isinstance(loss, list):
            assert len(loss) == 1
            loss = loss[0]
        return torch.mean(loss, dim=0)
    
    def give_encoder_model(self) -> Model:
        #function to return the encoder model
        #as a way to transport the pretrained encoder
        #into another learning context or saving the parameters manually
        return self.backbone
    
    def save_pretrained_model(self, save_path) -> None:
        model = self.backbone

        run_name = 'pretrained_model'

        save_path = os.path.join(save_path, run_name)
        print('saving to', save_path)
        os.makedirs(save_path, exist_ok=True)

        model.save_state_dict(f"{save_path}/state_dict.pth")
        model.save_config(f"{save_path}/model_config.yml")