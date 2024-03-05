import math

import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv

from modules import utils
from modules.cbam import CBAM


class GCNBlock(nn.Module):

    def __init__(self, num_edges, in_channels, out_channels):
        super(GCNBlock, self).__init__()
        hidden_channels = (in_channels+out_channels) // 2
        # learnable edge weights: https://github.com/pyg-team/pytorch_geometric/issues/2033#issuecomment-765210915
        self.edge_weight = torch.nn.Parameter(torch.ones(num_edges, dtype=torch.float32))
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels, eps=1e-6)
        self.act1 = nn.GELU()
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.norm2 = nn.LayerNorm(out_channels, eps=1e-6)
        self.act2 = nn.GELU()
        if in_channels != out_channels:
            self.downsample = nn.Sequential(*[
                nn.Linear(in_channels, out_channels, bias=False),
                nn.LayerNorm(out_channels, eps=1e-6)
            ])
        else:
            self.downsample = nn.Identity()

    def forward(self, x, edge_index):
        shortcut = x
        x = self.conv1(x, edge_index, self.edge_weight.sigmoid())
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x, edge_index, self.edge_weight.sigmoid())
        x = self.norm2(x)
        x += self.downsample(shortcut)
        x = self.act2(x)
        return x


class LocalBranch(nn.Module):
    """ LocalBranch is used to extract and enrich the representations of key positions in the image.
    """
    def __init__(self, depth, embed_dim, num_parts, part_channels, batch_size, num_patches, gaussian_ksize=None):
        super().__init__()

        if gaussian_ksize:
            self.kernel_size = gaussian_ksize  #11  # ablation
            kernel = utils.generate_gaussian_kernel(self.kernel_size, 7)
            kernel = torch.from_numpy(kernel).to(torch.float32).unsqueeze(0).unsqueeze(0)  # (out_channel,in_channel,kernel_size)
        else:
            self.kernel_size = None
        self.register_buffer('kernel', kernel)
        self.num_parts = num_parts
        self.part_channels = part_channels
        self.parts_dim = embed_dim // self.num_parts
        cbam_list = [CBAM(gate_channels=self.parts_dim, part_channels=self.part_channels) for i in range(self.num_parts)]
        self.cbam_list = nn.ModuleList(cbam_list)

        # init edges_index as complete graph: https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
        edge_index = [[], []]
        for i in range(self.num_parts):
            for j in range(self.num_parts):
                if i != j:
                    edge_index[0].append(i)
                    edge_index[1].append(j)
        edge_index = torch.tensor(edge_index, dtype=torch.long)  # (2,num_edges)
        # make edge_index compatible with batch size: https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html
        num_edges = edge_index.shape[1]
        batch_edge_index = []
        for i in range(batch_size):
            batch_edge_index.append(edge_index.clone()*i)
        edge_index = torch.cat(batch_edge_index, dim=1)  # (2,num_edges*batch_size)
        num_edges = edge_index.shape[1]
        self.register_buffer("edge_index", edge_index)
        self.batch_size = batch_size

        gcn_dim = 392
        if self.part_channels*num_patches != gcn_dim:
            self.before_gcn_fc = nn.Linear(self.part_channels*num_patches, gcn_dim)
        else:
            self.before_gcn_fc = nn.Identity()

        blocks = []
        self.num_channels = [gcn_dim]*depth + [embed_dim]
        for i in range(depth):
            block = GCNBlock(num_edges, in_channels=self.num_channels[i], out_channels=self.num_channels[i+1])
            blocks.append(block)
        self.local_backbone = nn.ModuleList(blocks)

    def locate_parts(self, decision_masks, x):
        with torch.no_grad():
            if self.kernel_size:
                decision_masks = utils.smooth_decision_mask(self.kernel, self.kernel_size, decision_masks)
            B, L, D = x.shape
            size = int(math.sqrt(L))
            x = x.permute(0,2,1).reshape(B, D, size, size)
            decision_masks = decision_masks.permute(0,2,1).reshape(B, 1, size, size)

        parts_masks = []
        x_list_in = torch.split(x, self.parts_dim, dim=1)  # Divide features into num_parts groups in channel dimension.
        x_list_out = []
        for x_in, cbam in zip(x_list_in, self.cbam_list):
            assert x_in.shape[1] == self.parts_dim, "x_in.shape[1] != self.parts_dim, {} != {}".format(x_in.shape[1], self.parts_dim)
            x_t, parts_mask = cbam(x_in, decision_masks)
            x_list_out.append(x_t)
            parts_masks.append(parts_mask)
        x = torch.cat(x_list_out, dim=1)  # B, num_parts*part_channels, H, W
        parts_masks = torch.cat(parts_masks, dim=1)  # B, num_parts, H, W
        return x, parts_masks

    def forward(self, decision_masks, x):
        shortcut = torch.mean(x, dim=1)  # B, L, D --> B, D
        B, L, D = x.shape

        x, parts_masks = self.locate_parts(decision_masks, x)  # B, D1, sqrt(L), sqrt(L)

        padding = False
        if x.shape[0] < self.batch_size:
            padding = True
        if padding:
            ori_batch_size = B
            x = torch.cat([x, torch.zeros((self.batch_size-x.shape[0],*x.shape[1:]), dtype=x.dtype, device=x.device)], dim=0)
            B = x.shape[0]
        x = x.view(B, self.num_parts, self.part_channels, L).reshape(B*self.num_parts, self.part_channels*L)  # num_nodes, part_channels; flatten batch_size dimension: https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html
        x = self.before_gcn_fc(x)
        for block in self.local_backbone:
            x = block(x, self.edge_index)
        x = x.view(B, self.num_parts, D).contiguous()  # B, L1, D
        if padding:
            x = x[:ori_batch_size]
        x = torch.mean(x, dim=1)  # B, D

        x += shortcut

        return x, parts_masks
