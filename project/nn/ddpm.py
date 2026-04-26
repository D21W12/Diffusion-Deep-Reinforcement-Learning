from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseEmbedding(nn.Module):

    def __init__(
            self,
            dim: int
    ):
        super().__init__()
        self._dim = dim

    def forward(
            self,
            sigmas: torch.Tensor,
    ):

        half_dim = self._dim // 2
        i = torch.arange(half_dim, device=sigmas.device)
        emb = 1 / 10000 ** (2 * i / self._dim)
        emb = torch.outer(sigmas, emb)
        return torch.concat([torch.sin(emb), torch.cos(emb)], dim=1)


class Downsample(nn.Module):

    def __init__(
            self,
            num_channels: int,
            use_conv: bool = True
    ):

        super().__init__()


        if use_conv:
            self._transform = nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                padding=1,
                stride=2
            )
        else:
            self._transform = nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            )

    def forward(self, X):
        return self._transform(X)


class Upsample(nn.Module):

    def __init__(
            self,
            num_channels: int,
            use_conv: bool = True
    ):

        super().__init__()


        if use_conv:
            self._conv = nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                padding='same',
                stride=1
            )
        else:
            self._conv = None

    def forward(self, X):

        y = F.interpolate(
            X,
            scale_factor=2
        )
        if self._conv:
            y = self._conv(y)
        return y


# TODO: fix attention
class Attention(nn.Module):

    def __init__(
            self,
            num_channels: int,
    ):

        super().__init__()

        self._norm = nn.GroupNorm(
            num_groups=32,
            num_channels=num_channels
        )

        # These are all network in network connections (1x1 convolutions)
        self._q = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=1,
            bias=True
        )
        self._k = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=1,
            bias=True
        )
        self._v = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=1,
            bias=True
        )

        self._conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, X):

        y = self._norm(X)

        q, k, v = self._q(y), self._k(y), self._v(y)

        y = F.scaled_dot_product_attention(q, k, v)
        y = self._conv(y)

        return X + y


class ResidualBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout: float = 0.,
            shortcut: Literal["conv", "nin"] = "nin"
    ):

        super().__init__()

        self._linear0 = nn.LazyLinear(
            out_features=out_channels
        )
        self._act0 = nn.SiLU()

        self._norm1 = nn.GroupNorm(
            num_groups=32,
            num_channels=in_channels,
        )
        self._act1 = nn.SiLU()
        self._conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding='same',
            stride=1,
            bias=True
        )
        self._norm2 = nn.GroupNorm(
            num_groups=32,
            num_channels=out_channels
        )
        self._act2 = nn.SiLU()
        self._drop = nn.Dropout(p=dropout)

        self._conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding='same',
            stride=1,
            bias=True
        )

        if in_channels != out_channels:
            if shortcut == "nin":
                self._conv3 = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding=0,
                    stride=1
                )
            else:
                self._conv3 = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding='same',
                    stride=1
                )
        else:
            self._conv3 = None

    def forward(
            self,
            X: torch.Tensor,
            embedding: torch.Tensor
    ) -> torch.Tensor:

        embedding = self._linear0(self._act0(embedding))

        h = self._act1(self._norm1(X))
        h = self._conv1(h)

        h += embedding[:, :, None, None]

        h = self._act2(self._norm2(h))
        h = self._drop(h)
        h = self._conv2(h)

        if self._conv3:
            X = self._conv3(X)

        return X + h


class UNet(nn.Module):
    """
    This UNet architecture follows the same architecture as Ho et al. (2020),
    translated directly from their tensorflow implementation.
    """

    def _get_encoder_channels(
            self,
            i_level: int,
            i_block: int | None = None,
            out: bool | None = None,
            attention: bool = False,
    ):
        assert out is not None or attention or i_block is None

        if i_block is None:
            return self._start_channels * self._channel_multipliers[i_level]
        elif attention:
            return self._start_channels * self._channel_multipliers[i_level]
        elif out:
            return self._start_channels * self._channel_multipliers[i_level]
        elif i_level == 0 and i_block == 0:
            return self._start_channels
        elif i_block == 0:
            return self._start_channels * self._channel_multipliers[i_level - 1]
        else:
            return self._start_channels * self._channel_multipliers[i_level]

    def _get_decoder_channels(
            self,
            i_level: int,
            i_block: int | None = None,
            out: bool | None = None,
            attention: bool = False,
    ):
        if i_block is None:
            return self._start_channels * self._channel_multipliers[i_level]
        elif attention:
            return self._start_channels * self._channel_multipliers[i_level]
        elif out:
            return self._start_channels * self._channel_multipliers[i_level]
        elif i_level == len(self._channel_multipliers) - 1 and i_block == 0:
            in_ = self._start_channels * self._channel_multipliers[-1]
            skip = self._get_skip_channels(i_level, i_block)
            return in_ + skip
        elif i_block == 0:
            in_ = self._start_channels * self._channel_multipliers[i_level + 1]
            skip = self._get_skip_channels(i_level, i_block)
            return in_ + skip
        else:
            in_ = self._start_channels * self._channel_multipliers[i_level]
            skip = self._get_skip_channels(i_level, i_block)
            return in_ + skip

    def _get_skip_channels(
            self,
            i_level: int,
            i_block: int,
    ):
        if i_level != 0 and i_block == self._num_res_blocks:
            return self._start_channels * self._channel_multipliers[i_level - 1]
        elif i_level == 0 and i_block == self._num_res_blocks:
            return self._start_channels
        else:
            return self._start_channels * self._channel_multipliers[i_level]

    def _get_bottleneck_channels(self):
        return self._start_channels * self._channel_multipliers[-1]

    def __init__(
            self,
            resolution: int,
            in_channels: int,
            start_channels: int,
            out_channels: int,
            num_res_blocks: int,
            channel_multipliers: set[int] | tuple[int],
            attention_resolutions: set[int] | tuple[int],
            dropout: float = 0.,
            conv_resample: bool = True
    ):

        assert len(channel_multipliers) > 0

        super().__init__()

        self._resolution = resolution
        self._in_channels = in_channels
        self._start_channels = start_channels
        self._out_channels = out_channels
        self._num_res_blocks = num_res_blocks
        self._channel_multipliers = channel_multipliers
        self._attention_resolutions = attention_resolutions

        # Initializing embeddings
        self._emb = nn.Sequential(
            NoiseEmbedding(dim=start_channels),
            nn.Linear(
                in_features=start_channels,
                out_features=start_channels * 4,
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=start_channels * 4,
                out_features=start_channels * 4,
            ),
        )

        # Start convolution
        self._start = nn.Conv2d(
            in_channels=in_channels,
            out_channels=start_channels,
            kernel_size=3,
            padding='same',
            stride=1,
            bias=True
        )

        # Initializing the encoder
        self._encoder = nn.ModuleList()
        for i_level in range(len(channel_multipliers)):

            encoder_block = nn.ModuleDict()
            encoder_block["residuals"] = nn.ModuleList()
            for i_block in range(num_res_blocks):

                block = nn.ModuleDict()
                block["residual"] = ResidualBlock(
                    in_channels=self._get_encoder_channels(i_level, i_block, out=False),
                    out_channels=self._get_encoder_channels(i_level, i_block, out=True),
                    dropout=0
                )

                if resolution // 2 ** i_level in attention_resolutions:
                    block["attention"] = Attention(
                        num_channels=self._get_encoder_channels(i_level, i_block, attention=True)
                    )

                encoder_block["residuals"].append(block)


            if i_level != len(channel_multipliers) - 1:

                encoder_block["downsample"] = Downsample(
                    num_channels=self._get_encoder_channels(i_level),
                    use_conv=conv_resample
                )

            self._encoder.append(encoder_block)

        bottleneck_num = self._get_bottleneck_channels()

        # Initializing the bottleneck
        self._bottleneck_res1 = ResidualBlock(
            in_channels=bottleneck_num,
            out_channels=bottleneck_num,
            dropout=0
        )
        self._bottleneck_attn = Attention(
            num_channels=bottleneck_num
        )
        self._bottleneck_res2 = ResidualBlock(
            in_channels=bottleneck_num,
            out_channels=bottleneck_num,
            dropout=0
        )

        # Initializing the decoder
        self._decoder = nn.ModuleList()
        for i_level in range(len(channel_multipliers)):

            decoder_block = nn.ModuleDict()
            decoder_block["residuals"] = nn.ModuleList()
            for i_block in range(num_res_blocks + 1):

                block = nn.ModuleDict()
                block["residual"] = ResidualBlock(
                    in_channels=self._get_decoder_channels(i_level, i_block, out=False),
                    out_channels=self._get_decoder_channels(i_level, i_block, out=True),
                    dropout=dropout
                )

                if resolution // 2 ** i_level in attention_resolutions:

                    block["attention"] = Attention(
                        num_channels=self._get_decoder_channels(i_level, i_block, attention=True)
                    )

                decoder_block["residuals"].append(block)

            if i_level != 0:
                decoder_block["upsample"] = Upsample(
                    num_channels=self._get_decoder_channels(i_level),
                    use_conv=conv_resample
                )

            self._decoder.append(decoder_block)

        # End convolution
        self._end = nn.Sequential(
            nn.GroupNorm(
                num_groups=32,
                num_channels=start_channels * channel_multipliers[0]
            ),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=start_channels * channel_multipliers[0],
                out_channels=out_channels,
                kernel_size=3,
                padding='same',
                stride=1,
                bias=True
            )
        )

    def forward(self, X, sigmas):

        # Type annotations for for-loops
        encoder_block: nn.ModuleDict[str, nn.ModuleList | nn.Module]
        decoder_block: nn.ModuleDict[str, nn.ModuleList | nn.Module]
        res_block: nn.ModuleDict

        emb = self._emb(sigmas)

        hs = [self._start(X)]

        # Encoder
        for encoder_block in self._encoder:
            for res_block in encoder_block["residuals"]:
                h = res_block["residual"](hs[-1], emb)
                if "attention" in res_block:
                    h = res_block["attention"](h)
                hs.append(h)
            if "downsample" in encoder_block:
                hs.append(encoder_block["downsample"](hs[-1]))

        h = hs[-1]

        # Bottleneck
        h = self._bottleneck_res1(h, emb)
        h = self._bottleneck_attn(h)
        h = self._bottleneck_res2(h, emb)

        # Decoder
        for decoder_block in reversed(self._decoder):
            for res_block in decoder_block["residuals"]:
                h = torch.concat([h, hs.pop()], dim=1)
                h = res_block["residual"](h, emb)
                if "attention" in res_block:
                    h = res_block["attention"](h)
            if "upsample" in decoder_block:
                h = decoder_block["upsample"](h)

        return self._end(h)


if __name__ == "__main__":
    model = UNet(
        resolution=32,
        in_channels=3,
        start_channels=128,
        out_channels=3,
        num_res_blocks=1,
        channel_multipliers=(2, 2, 2, 2),
        attention_resolutions=[4],
    )
    x = torch.zeros((16, 3, 32, 32))
    sigmas = torch.linspace(0.2, 80, 16)
    y = model(x, sigmas)
