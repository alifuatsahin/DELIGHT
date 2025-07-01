from model.point import Point

import torch


if __name__ == "__main__":
    coords = torch.rand(2, 10, 3) * 10

    B, N, C = coords.shape

    pcs = coords.view(B * N, C)
    batch = torch.arange(B, device=coords.device).view(-1, 1).repeat(1, N).view(-1)

    print("Coordinates shape:", coords)
    print("Batch shape:", batch.shape)
    print("Batch:", batch)
    print("coords:", coords)
    print("PCS:", pcs)

    # grid_size = 0.5

    # data_dict = {
    #     "coord": coords,
    #     "grid_size": grid_size,
    # }
    # point = Point(data_dict)
    # point.serialization(order=("z", "z-trans", "hilbert", "hilbert-trans"))
    # # print("Original coordinates:\n", point.coord)
    # # print("Grid coordinates:\n", point.grid_coord)
    # # print("Order:\n", point.serialized_order)
    # # print("Inverse order:\n", point.serialized_inverse)

    # shuffle_orders = True
    # code = point.code
    # order = point.serialized_order
    # inverse = point.serialized_inverse

    # print(code.shape)

    # if shuffle_orders:
    #     perm = torch.randperm(code.shape[0])
    #     code = code[perm]
    #     order = order[perm]
    #     inverse = inverse[perm]
