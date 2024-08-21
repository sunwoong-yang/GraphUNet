from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.nn.pool.connect import FilterEdges
# from torch_geometric.nn.pool.select import SelectTopK
from torch_geometric.typing import OptTensor
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.pool.select import Select, SelectOutput
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import cumsum, scatter, softmax

# This code is modified from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/pool/topk_pool.py
class TopKPooling(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        ratio: Union[int, float] = 0.5,
        min_score: Optional[float] = None,
        multiplier: float = 1.,
        nonlinearity: Union[str, Callable] = 'tanh',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier

        self.select = SelectTopK(in_channels, ratio, min_score, nonlinearity)
        self.connect = FilterEdges()

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.select.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        attn: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, OptTensor, OptTensor, Tensor, Tensor]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The node feature matrix.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example. (default: :obj:`None`)
            attn (torch.Tensor, optional): Optional node-level matrix to use
                for computing attention scores instead of using the node
                feature matrix :obj:`x`. (default: :obj:`None`)
        """
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        select_out = self.select(attn, batch)

        perm = select_out.node_index
        score = select_out.weight
        # if score is None:
        #     score = 1 # if weight of the returned value by self.select is None, it means gvPool is being applied and therefore x is not being multiplied by the score
        assert score is not None

        x = x[perm] * score.view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        connect_out = self.connect(select_out, edge_index, edge_attr, batch)

        return (x, connect_out.edge_index, connect_out.edge_attr,
                connect_out.batch, perm, score)
	#
    # def __repr__(self) -> str:
    #     if self.min_score is None:
    #         ratio = f'ratio={self.ratio}'
    #     else:
    #         ratio = f'min_score={self.min_score}'
	#
    #     return (f'{self.__class__.__name__}({self.in_channels}, {ratio}, '
    #             f'multiplier={self.multiplier})')

def topk(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
    min_score: Optional[float] = None,
    tol: float = 1e-7,
) -> Tensor:
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)
        return perm

    if ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0), ), int(ratio))
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        x, x_perm = torch.sort(x.view(-1), descending=True)
        batch = batch[x_perm]
        batch, batch_perm = torch.sort(batch, descending=False, stable=True)

        arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        ptr = cumsum(num_nodes)
        batched_arange = arange - ptr[batch]
        mask = batched_arange < k[batch]

        return x_perm[batch_perm[mask]]

    raise ValueError("At least one of the 'ratio' and 'min_score' parameters "
                     "must be specified")


class SelectTopK(Select):
    r"""Selects the top-:math:`k` nodes with highest projection scores from the
    `"Graph U-Nets" <https://arxiv.org/abs/1905.05178>`_, `"Towards Sparse
    Hierarchical Graph Classifiers" <https://arxiv.org/abs/1811.01287>`_
    and `"Understanding Attention and Generalization in Graph Neural
    Networks" <https://arxiv.org/abs/1905.02850>`_ papers.

    If :obj:`min_score` :math:`\tilde{\alpha}` is :obj:`None`, computes:

        .. math::
            \mathbf{y} &= \sigma \left( \frac{\mathbf{X}\mathbf{p}}{\|
            \mathbf{p} \|} \right)

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

    If :obj:`min_score` :math:`\tilde{\alpha}` is a value in :obj:`[0, 1]`,
    computes:

        .. math::
            \mathbf{y} &= \mathrm{softmax}(\mathbf{X}\mathbf{p})

            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

    where :math:`\mathbf{p}` is the learnable projection vector.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float or int): The graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`.
            This value is ignored if :obj:`min_score` is not :obj:`None`.
            (default: :obj:`0.5`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        act (str or callable, optional): The non-linearity :math:`\sigma`.
            (default: :obj:`"tanh"`)
    """
    def __init__(
        self,
        in_channels: int,
        ratio: Union[int, float] = 0.5,
        min_score: Optional[float] = None,
        act: Union[str, Callable] = 'tanh',
    ):
        super().__init__()

        if ratio is None and min_score is None:
            raise ValueError(f"At least one of the 'ratio' and 'min_score' "
                             f"parameters must be specified in "
                             f"'{self.__class__.__name__}'")

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.act = activation_resolver(act)

        if in_channels == 1:
            self.weight = torch.nn.Parameter(torch.empty(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        if self.in_channels == 1:
            uniform(self.in_channels, self.weight)

    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
    ) -> SelectOutput:
        """"""  # noqa: D419
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # var_x = torch.var(x, dim=1)
        x = x.view(-1, 1) if x.dim() == 1 else x
        # 여기서부터 수정 필요 X*p 대신 X*var(X)로 수정하면됨
        # my_weight = var_x #.reshape(1, self.in_channels)
        # print('asdf', var_x.shape, x.shape, self.weight.shape)

        if x.shape[1] != 1:
            score = torch.var(x, dim=1)
            # Since back-prop is possible, no need for projection (X*y)
        else:
            score = (x * self.weight).sum(dim=-1)
            if self.min_score is None:
                score = self.act(score / self.weight.norm(p=2, dim=-1))
            else:
                score = softmax(score, batch)
        # print('scor1',score.shape)
        # score = torch.var(x, dim=1)
        # print('kkkkkkkkkkkkkkkk')
        # if self.min_score is None:
        #     score = self.act(score / self.weight.norm(p=2, dim=-1))
        # else:
        #     score = softmax(score, batch)

        node_index = topk(score, self.ratio, batch, self.min_score)

        if x.shape[1] != 1:
            return SelectOutput(
                node_index=node_index,
                num_nodes=x.size(0),
                cluster_index=torch.arange(node_index.size(0), device=x.device),
                num_clusters=node_index.size(0),
                weight=torch.ones_like(score[node_index]), # when gvPool, weighting of X with score is not required
            )
        else:
            return SelectOutput(
                node_index=node_index,
                num_nodes=x.size(0),
                cluster_index=torch.arange(node_index.size(0), device=x.device),
                num_clusters=node_index.size(0),
                weight=score[node_index],
            )

    def __repr__(self) -> str:
        if self.min_score is None:
            arg = f'ratio={self.ratio}'
        else:
            arg = f'min_score={self.min_score}'
        return f'{self.__class__.__name__}({self.in_channels}, {arg})'