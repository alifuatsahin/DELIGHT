from model.point import Point

from model.encoder import Encoder

import torch


if __name__ == "__main__":
    enc = Encoder()
    input = torch.randn(3, 1024, 3)  # Example input tensor

    output = enc(input)

    print(output.feat)