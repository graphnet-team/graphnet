#important: need to install flash attention; follow guide on https://github.com/Dao-AILab/flash-attention
#for me I needed to first conda install nvcc
from typing import Set, Union, List, Type, Optional, Dict, Any
import os

import torch
from torch.optim.adam import Adam
from torch.functional import Tensor

from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import knn_graph
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.utils import mask_select
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

from torch_scatter import scatter

from pytorch_lightning import LightningModule

from flash_attn.modules.mha import MHA

from graphnet.models import Model
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.gnn.gnn import GNN
from graphnet.models.easy_model import EasySyntax
from graphnet.models.task import IdentityTask
from graphnet.models.components.layers import DropPath

from graphnet.training.loss_functions import MSELoss


##Block with flash_attn/Theseus model##
#region

class flash_Mlp(LightningModule):
    """Multi-Layer Perceptron (MLP) module."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: torch.nn.Module = torch.nn.GELU,
        dropout_prob: float = 0.0,
    ):
        """Construct `Mlp`.

        This is mostly analogous to the Mlp for DeepIce other than the dtypes being chnaged to torch.float16

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden features. Defaults to None.
                If None, it is set to the value of `in_features`.
            out_features: Number of output features. Defaults to None.
                If None, it is set to the value of `in_features`.
            activation: Activation layer. Defaults to `nn.GELU`.
            dropout_prob: Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        if in_features <= 0:
            raise ValueError(
                f"in_features must be greater than 0, got in_features "
                f"{in_features} instead"
            )
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.input_projection = torch.nn.Linear(in_features, hidden_features, dtype=torch.bfloat16)
        self.activation = activation()
        self.output_projection = torch.nn.Linear(hidden_features, out_features, dtype=torch.bfloat16)
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.output_projection(x)
        x = self.dropout(x)
        return x

class rope_embedder(Model):
    def __init__(self,
                 token_dim=384,
                 max_seqlen=2560,
                 ):
        super().__init__()
        if token_dim % 2 != 0:
            new_token_dim = token_dim + 1
        else:
            new_token_dim = token_dim

        self.lin = torch.nn.Sequential(torch.nn.Linear(token_dim, token_dim),
                                       torch.nn.SELU(),
                                       torch.nn.BatchNorm1d(token_dim),
                                       torch.nn.Linear(token_dim, new_token_dim),
                                       torch.nn.SELU())
        
        exp = -2*(torch.arange(1, new_token_dim/2 + 1).repeat_interleave(2) -1)/new_token_dim
        base = 10000
        theta_mat = torch.arange(1,max_seqlen+1).view(-1,1).expand(-1,new_token_dim)*torch.pow(base, exp)
        self.cos_theta_mat = torch.cos(theta_mat)
        self.sin_theta_mat = torch.sin(theta_mat)


    def forward(self, data:Data):
        x_hat = self.lin(data.x)
        batched_x, mask = to_dense_batch(x=x_hat, batch=data.batch)
        emb = batched_x*self.cos_theta_mat[0:batched_x.shape[1]].to(device=self.device)
        indeces_even = torch.arange(0,batched_x.shape[2],2).to(device=self.device)
        indeces_odd = torch.arange(1,batched_x.shape[2],2).to(device=self.device)
        batched_x[:,:,indeces_odd], batched_x[:,:,indeces_even] = -batched_x[:,:,indeces_even], batched_x[:,:,indeces_odd]
        emb += batched_x*self.sin_theta_mat[0:batched_x.shape[1]].to(device=self.device)
        return emb[mask]
    
class flashMHA_block(LightningModule):
    """Implementation of BEiTv2 Block."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        softm_scale: Optional[float] = None,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        activation: torch.nn.Module = torch.nn.GELU,
        norm_layer: torch.nn.Module = torch.nn.LayerNorm,
    ):
        """Construct 'Block_rel'.

        Implements flash_attn's MHA module. Most of the arguments pertain to everything but theís module and as of now
        tweaking the actual attention mechanism can be done by adjusting the parameters passed to the MHA in the class initialization
        by hand.

        Args:
            input_dim: Dimension of the input tensor.
            num_heads: Number of attention heads to use in the `Attention_rel`
            layer.
            mlp_ratio: Ratio of the hidden size of the feedforward network to
                the input size in the `Mlp` layer.
            qkv_bias: Whether or not to include bias terms in the query, key,
                and value matrices in the `Attention_rel` layer.
            qk_scale: Scaling factor for the dot product of the query and key
                matrices in the `Attention_rel` layer.
            dropout: Dropout probability to use in the `Mlp` layer.
            attn_drop: Dropout probability to use in the `Attention_rel` layer.
            drop_path: Probability of applying drop path regularization to the
                output of the layer.
            init_values: Initial value to use for the `gamma_1` and `gamma_2`
                parameters if not `None`.
            activation: Activation function to use in the `Mlp` layer.
            norm_layer: Normalization layer to use.
            attn_head_dim: Dimension of the attention head outputs in the
                `Attention_rel` layer.
        """
        super().__init__()
        self.norm1 = norm_layer(input_dim, dtype=torch.bfloat16, eps=1e-05, elementwise_affine=True)
        self.attn = MHA(embed_dim=input_dim,
                        num_heads=num_heads,
                        use_flash_attn=True,
                        softmax_scale=softm_scale,
                        dropout=attn_drop,
                        dtype=torch.bfloat16)
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()
        )
        self.norm2 = norm_layer(input_dim, dtype=torch.bfloat16, eps=1e-05, elementwise_affine=True)
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = flash_Mlp(
            in_features=input_dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            dropout_prob=dropout,
        )

        if init_values is not None:
            self.gamma_1 = torch.nn.Parameter(
                init_values * torch.ones(input_dim).to(device=self.device), requires_grad=True
            ).to(device=self.device, dtype=torch.bfloat16)
            self.gamma_2 = torch.nn.Parameter(
                init_values * torch.ones(input_dim).to(device=self.device), requires_grad=True
            ).to(device=self.device, dtype=torch.bfloat16)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(
        self,
        x: Tensor,
        cu_seqs: Tensor,
        max_seq: int,


    ) -> Tensor:
        """Forward pass."""
        if self.gamma_1 is None:
            xn = self.norm1(x)
            x = x + self.drop_path(
                self.attn(
                    xn,
                    cu_seqlens=cu_seqs,
                    max_seqlen=max_seq,
                )
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            xn = self.norm1(x)
            x = x + self.drop_path(
                self.gamma_1.to(device=self.device)
                * self.drop_path(
                    self.attn(
                        xn,
                        cu_seqlens=cu_seqs,
                        max_seqlen=max_seq,
                    )
                )
            )
            x = x + self.drop_path(self.gamma_2.to(device=self.device) * self.mlp(self.norm2(x)))
        return x

class Theseus_DeepIce(GNN):
    """DeepIce model."""

    def __init__(
        self,
        compression_model,   #here the variables to the compression are given
        fix_compression = False,
        max_length = 2560,
        hidden_dim: int = 384,   #here the original TheseusDeepIce variables are given (possibly with some changes)
        mlp_ratio: int = 4,
        depth_two: int = 12,
        head_size: int = 32,
        depth_one: int = 4,
        include_dynedge: bool = False,
        dynedge_args: Optional[Dict[str, Any]] = None,
    ):
        """Construct `DeepIce`.

        Args:
            hidden_dim: The latent feature dimension.
            mlp_ratio: Mlp expansion ratio of FourierEncoder and Transformer.
            seq_length: The base feature dimension.
            depth: The depth of the transformer.
            head_size: The size of the attention heads.
            depth_rel: The depth of the relative transformer.
            n_rel: The number of relative transformer layers to use.
            scaled_emb: Whether to scale the sinusoidal positional embeddings.
            include_dynedge: If True, pulse-level predictions from `DynEdge`
                will be added as features to the model.
            dynedge_args: Initialization arguments for DynEdge. If not
                provided, DynEdge will be initialized with the original Kaggle
                Competition settings. If `include_dynedge` is False, this
                argument have no impact.
            n_features: The number of features in the input data.
            have_rel_bias: choose whether to use rel_pos_bias or not.
                False is recommended for training with a compression method
        """
        super().__init__(max_length, hidden_dim)

        self.compression = compression_model
        
        self.fix = fix_compression

        self.embedding = rope_embedder(token_dim=hidden_dim, max_seqlen=max_length)
        
        #here can choose between simple MHA block or unpacked qkv with flashv2_block
        self.sandwich = torch.nn.ModuleList(
            [
                flashMHA_block(
                    input_dim=hidden_dim,
                    num_heads=hidden_dim // head_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=0.0 * (i / (depth_one - 1)),
                    init_values=1,
                )
                for i in range(depth_one)
            ]
        )

        self.cls_token = torch.nn.Linear(hidden_dim, 1, bias=False, dtype=torch.bfloat16)
        self.blocks = torch.nn.ModuleList(
            [
                flashMHA_block(
                    input_dim=hidden_dim,
                    num_heads=hidden_dim // head_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=0.0 * (i / (depth_two - 1)),
                    init_values=1,
                )
                for i in range(depth_two)
            ]
        )

        #dynedge specifications
        if include_dynedge and dynedge_args is None:
            self.warning_once("Running with default DynEdge settings")
            self.dyn_edge = DynEdge(
                nb_inputs=9,
                nb_neighbours=9,
                post_processing_layer_sizes=[336, hidden_dim // 2],
                dynedge_layer_sizes=[
                    (128, 256),
                    (336, 256),
                    (336, 256),
                    (336, 256),
                ],
                global_pooling_schemes=None,
                activation_layer="gelu",
                add_norm_layer=True,
                skip_readout=True,
            )
        elif include_dynedge and not (dynedge_args is None):
            self.dyn_edge = DynEdge(**dynedge_args)

        self.include_dynedge = include_dynedge


    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """cls_tocken should not be subject to weight decay during training."""
        return {"cls_token"}

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""

        #run through the compression model and embedding
        if self.fix:
            with torch.no_grad():
                compressed_data = self.compression(data)
        else:
            compressed_data = self.compression(data)

        x = self.embedding(compressed_data).to(dtype=torch.bfloat16)
        #print(x.shape)
        compressed_batch = compressed_data.batch

        #auxiliary variables for the Transformer architectures
        _,seq_lengths = torch.unique_consecutive(compressed_batch, return_counts=True)
        batch_size = seq_lengths.shape[0]
        cu_seqs = torch.nn.functional.pad(seq_lengths.cumsum(0), pad=(1,0), value=0).to(torch.int32)
        max_seq = seq_lengths.max().to(device=self.device, dtype=torch.int32).item()

        #dynedge inclusion as in DeepIce
        if self.include_dynedge:
            #test dynedge inclusion at later point!!!
            graph = self.dyn_edge(data)
            x = torch.cat([x, graph], 1)
 
        #actual Transformer procedure
        for blk in self.sandwich:
            x = blk(x=x,
                    cu_seqs=cu_seqs,
                    max_seq=max_seq,)

        cls_token = self.cls_token.weight.expand(
           batch_size, -1
        )
        #create empty tensor and append cls_tokens at the beginning of the individual sequences; fill the rest with original x
        emp = torch.empty((x.shape[0]+batch_size, x.shape[1])).to(device=self.device, dtype=torch.bfloat16)
        cls_ind = cu_seqs + torch.arange(batch_size+1).to(device=self.device, dtype=torch.int32)
        ran = torch.arange(x.shape[0]+batch_size).to(device=self.device)
        emp[cls_ind[:-1],:] = cls_token
        emp[ran[~torch.isin(ran,cls_ind[:-1]).to(device=self.device)],:] = x
        x=emp

        for blk in self.blocks:
            x = blk(x=x,
                    cu_seqs=cls_ind,
                    max_seq=max_seq+1,)

        return x[cls_ind[:-1], :].to(dtype=torch.float32)
    
#endregion

##Block with mask_pred pretraining##
#region

def dense_mse_loss(reco, orig, bv):
    squared_errs = (reco - orig)**2
    losses = torch.mean(scatter(src=squared_errs, index=bv, reduce='mean', dim=0), dim=0)

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
                 nb_mp: int = 1,
                 ):
        super().__init__()

        self.activation = torch.nn.SELU()
        
        self.lin_net = torch.nn.ModuleList()
        for i in range(nb_linear):
            if i == 0:
                self.lin_net.append(torch.nn.Linear(in_dim,hidden_dim))
            else:
                self.lin_net.append(torch.nn.Linear(hidden_dim,hidden_dim))

        self.conv_net = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(nb_mp):
            self.conv_net.append(GCNConv(in_channels=hidden_dim, out_channels=hidden_dim, add_self_loops=False))
            self.norms.append(torch.nn.BatchNorm1d(hidden_dim))

        self.final_proj = torch.nn.Linear(hidden_dim, out_dim)
        

    def forward(self, data:Union[Data, Tensor]):
        if isinstance(data, Data):
            x_hat = data.x
        else:
            x_hat = data
        x_hat = self.lin_net[0](x_hat)
        x_hat = self.activation(x_hat)
        for i in range(1,len(self.lin_net)):
            x_hat = x_hat + self.lin_net[i](x_hat)
            x_hat = self.activation(x_hat)
        for i in range(len(self.conv_net)):
            x_hat = x_hat + self.conv_net[i](x_hat, data.edge_index)
            x_hat = self.activation(x_hat)
            x_hat = self.norms[i](x_hat)

        x_hat = self.final_proj(x_hat)
        
        return x_hat

class mask_pred_augment(Model):
    def __init__(self, 
                 masked_ratio: float = 0.25,
                 masked_feat: List[int] = [0,1,2,3,4],
                 hlc_pos: int = None,):
        super().__init__()
        self.ratio = masked_ratio
        self.hlc_pos = hlc_pos
        self.masked_feat = masked_feat

    def forward(self, data: Data):
        auged = data.clone()

        rand_score = torch.rand_like(data.batch.to(dtype=torch.bfloat16))
        if self.hlc_pos is not None:
            rand_score = rand_score + auged.x[:,self.hlc_pos].view(1,-1)

        ind = topk(x=rand_score, ratio=self.ratio, batch=data.batch)

        mask = torch.ones_like(data.batch.to(dtype=torch.bfloat16))
        mask[ind] = 0

        target = mask_select(src=auged.x, dim=0, mask=~mask.bool())[:,self.masked_feat]
        auged.x[:,self.masked_feat] = auged.x[:,self.masked_feat]*mask.view(-1,1)

        return auged, target

class mask_pred_frame(EasySyntax):
    def __init__(self,
                 encoder: Model,
                 encoder_out_dim: int = None,
                 masked_ratio: float = 0.25,
                 masked_feat: List[int] = [0,1,2,3,4],
                 hlc_pos: int = None,
                 mask_pred_net: Model = None,
                 default_hidden_dim: int = 1000, 
                 default_nb_linear: int = 5,
                 default_nb_mp: int = 1,
                 final_loss: str = 'cosine',
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

        self.ratio = masked_ratio

        self.augment = mask_pred_augment(masked_ratio=masked_ratio,
                                         masked_feat=masked_feat,
                                         hlc_pos=hlc_pos)

        self.encoder = encoder

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
                                             nb_linear=default_nb_linear,
                                             nb_mp=default_nb_mp)
        else:
            assert mask_pred_net.nb_outputs == len(masked_feat), f'make sure that your \"mask_pred_net\" has number of output feats equal to nb of masked feats ({len(masked_feat)})'
            self.rep = mask_pred_net

        self.scorer = standard_maskpred_net(in_dim=len(masked_feat),
                                            hidden_dim=default_hidden_dim,
                                            out_dim=1,
                                            nb_linear=default_nb_linear,
                                            nb_mp=0)
        
        assert final_loss in ['cosine', 'mse'], f'can only choose from {['cosine', 'mse']} for loss function'
        if final_loss == 'cosine':
            self.loss_func = neg_cosine_loss
        elif final_loss == 'mse':
            self.loss_func = dense_mse_loss
            
    def forward(self, data: Union[Data, List[Data]]):
        if not isinstance(data, Data):
            data = data[0]

        aug, target = self.augment(data)

        data_hat = self.encoder(aug)

        rep = self.rep(data_hat)
        score = self.scorer(rep)

        ind = topk(x=score, ratio=self.ratio, batch=data.batch)

        nodes = rep[ind]*torch.sigmoid(score[ind])

        loss = self.loss_func(reco=nodes, orig=target, bv=data.batch[ind])

        #loss is returned as a list to comply with the graphnet predict functionality
        return [loss]

    def validate_tasks(self) -> None:
        accepted_tasks = IdentityTask
        for task in self._tasks:
            assert isinstance(task, accepted_tasks)

    def shared_step(self, batch: List[Data], batch_idx: int):
        loss = self(batch)
        if isinstance(loss, list):
            assert len(loss) == 1
            loss = loss[0]
        return torch.mean(loss, dim=0)
    
    def give_encoder_model(self):
        #function to return the encoder model
        #as a way to transport the pretrained encoder
        #into another learning context or saving the parameters manually
        return self.encoder
    
    def save_pretrained_model(self, save_path):
        model = self.encoder

        run_name = 'pretrained_model'

        save_path = os.path.join(save_path, run_name)
        print('would save to', save_path)
        #os.makedirs(save_path, exist_ok=True)

        #model.save_state_dict(f"{save_path}/state_dict.pth")
        #model.save_config(f"{save_path}/model_config.yml")
    
#endregion