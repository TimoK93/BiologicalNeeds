"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

"""
Modified by Timo Kaiser
"""

import torch
import torch.nn as nn
import embedtrack.models.erfnet as erfnet


class BranchedERFNet(nn.Module):
    def __init__(self, num_classes, input_channels=1, encoder=None):
        super().__init__()
        print("Creating branched erfnet with {} classes".format(num_classes))
        if encoder is None:
            self.encoder = erfnet.Encoder(sum(num_classes), input_channels)
        else:
            self.encoder = encoder
        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(erfnet.Decoder(n))

    def forward(self, input):
        output = self.encoder(input)
        return torch.cat(
            [decoder.forward(output) for decoder in self.decoders], 1)
