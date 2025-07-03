import sys
import torch
from addict import Dict
from collections import OrderedDict
import spconv.pytorch as spconv
import torch.nn as nn

from model.utils import offset2batch, batch2offset
from serialization import encode

class Point(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)
        

    def serizalization(self, order="z", depth=None, shuffle_orders=False):
        assert "batch" in self.keys()

        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord", "batch"}.issubset(self.keys())
            grid_coord = torch.empty_like(self.coord, dtype=torch.int32)
            for b in torch.unique(self.batch):
                idx = (self.batch == b)
                min_coord = self.coord[idx].min(0)[0]
                grid_coord[idx] = torch.div(
                    self.coord[idx] - min_coord, self.grid_size, rounding_mode="trunc"
                ).int()
            self["grid_coord"] = grid_coord

        if depth is None:
            depth = int(self.grid_coord.max()).bit_length()
        
        self["serialized_depth"] = depth
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 64, "Depth is too large for serialization."
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) mm^3
        # cube with a grid size of 0.01 mm. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.
        assert depth <= 16, "Depth is too large for serialization."

        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]
        
        self["serialized_code"] = code
        self["serialized_order"] = code
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        assert {"feat", "batch"}.issubset(self.keys())

        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord", "batch"}.issubset(self.keys())
            grid_coord = torch.empty_like(self.coord)
            for b in torch.unique(self.batch):
                idx = (self.batch == b)
                min_coord = self.coord[idx].min(0)[0]
                grid_coord[idx] = torch.div(
                    self.coord[idx] - min_coord, self.grid_size, rounding_mode="trunc"
                ).int()
            self["grid_coord"] = grid_coord

        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()]
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat

# class Point(Dict):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def serialization(self, order="z", depth=None, shuffle_orders=False):
#         if "grid_coord" not in self.keys():
#             assert {"grid_size", "coord"}.issubset(self.keys())
#             self["grid_coord"] = torch.div(
#                 self.coord - self.coord.min(dim=1, keepdim=True)[0], self.grid_size, rounding_mode="trunc"
#             ).int()

#         if depth is None:
#             depth = int(self.grid_coord.max()).bit_length()

#         self["serialized_depth"] = depth
#         # Maximum bit length for serialization code is 63 (int64)
#         assert depth * 3 + len(self.offset).bit_length() <= 64, "Depth is too large for serialization."

#         code = [
#             torch.stack(
#                 [torch.as_tensor(encode(grid_coord_, depth=depth, order=order_), device=self.grid_coord.device)
#                 for grid_coord_ in self.grid_coord]
#             )
#             for order_ in order
#         ]

#         print(code)
#         code = torch.stack(code)
#         order = torch.argsort(code)
#         inverse = torch.zeros_like(order).scatter_(
#             dim=1,
#             index=order,
#             src=torch.arange(0, code.shape[1], device=order.device).repeat(
#                 code.shape[0], 1
#             ),
#         )

#         self["code"] = code

#         if shuffle_orders:
#             perm = torch.randperm(code.shape[0])
#             code = code[perm]
#             order = order[perm]
#             inverse = inverse[perm]

#         self["serialized_code"] = code
#         self["serialized_order"] = order
#         self["serialized_inverse"] = inverse

#     def sparsify(self, pad=96):
#         if "grid_coord" not in self.keys():
#             assert {"grid_size", "coord"}.issubset(self.keys())
#             self["grid_coord"] = torch.div(
#                 self.coord - self.coord.min(dim=1, keepdim=True)[0], self.grid_size, rounding_mode="trunc"
#             ).int()

#         if "sparse_shape" in self.keys():
#             sparse_shape = self.sparse_shape
#         else:
#             sparse_shape = torch.add(
#                 torch.max(self.grid_coord, dim=0).values, pad
#             ).tolist()
#         sparse_conv_feat = spconv.SparseConvTensor(
#             features=self.feat,
#             indices=torch.cat(
#                 [self.batch.unsqueeze(-1).int(), self.grid_coord.int()]
#             ).contiguous(),
#             spatial_shape=sparse_shape,
#             batch_size=self.batch[-1].tolist() + 1,
#         )
#         self["sparse_shape"] = sparse_shape
#         self["sparse_conv_feat"] = sparse_conv_feat

class PointModule(nn.Module):
    """
    Base class for point modules, inheriting from nn.Module.
    This class is designed to be extended by specific point modules
    """
    def __init__(self):
        super().__init__()

class PointSequential(PointModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs are only supported in Python 3.6+")
            if name in self._modules:
                raise ValueError(f"Module {name} already exists in the model.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("Index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for _ in range(idx):
            next(it)
        return next(it)
    
    def __len__(self):
        return len(self._modules)
    
    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("Module name {} already exists.".format(name))
        self.add_module(name, module)

    def forward(self, input):
        for _, module in self._modules.items():
            if isinstance(module, PointModule):
                input = module(input)
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] == 0:
                        input = input.replace_feature(
                            module(input.features)
                        )
                else:
                    input = module(input)

        return input