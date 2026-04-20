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
        i = torch.arange(half_dim)
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
                padding='same',
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

        self._q = nn.LazyConv2d(
            out_channels=num_channels,
            kernel_size=1,
            bias=True
        )
        self._k = nn.LazyConv2d(
            out_channels=num_channels,
            kernel_size=1,
            bias=True
        )
        self._v = nn.LazyConv2d(
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
            dropout: float,
    ):

        super().__init__()

        self._linear0 = nn.Linear(
            in_features=in_channels,
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
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding='same',
            stride=1,
            bias=True
        )

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

        return X + h


class UNet(nn.Module):
    """
    This UNet architecture follows the same architecture as Ho et al. (2020),
    translated directly from their tensorflow implementation.
    """

    def __init__(
            self,
            resolution: int,
            in_channels: int,
            out_channels: int,
            num_res_blocks: int,
            channel_multipliers: set[int] | tuple[int],
            attention_resolutions: set[int] | tuple[int],
            dropout: float = 0
    ):

        super().__init__()

        channels = [in_channels] + [in_channels * m for m in channel_multipliers]

        # Initializing embeddings
        self._emb = nn.Sequential(
            NoiseEmbedding(dim=in_channels),
            nn.Linear(
                in_features=in_channels,
                out_features=in_channels * 4,
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=in_channels * 4,
                out_features=in_channels * 4,
            )
        )

        # Start convolution
        self._start = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
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

                block_in = channels[i_level + 1] if i_block != 0 else channels[i_level]
                block_out = channels[i_level + 1]

                block = nn.ModuleDict()
                block["residual"] = ResidualBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    dropout=0
                )

                if resolution // 2 ** i_block in attention_resolutions:
                    block["attention"] = Attention(
                        num_channels=block_out
                    )

                encoder_block["residuals"].append(block)

            if i_level != len(channel_multipliers) - 1:
                encoder_block["downsample"] = Downsample(
                    num_channels=channels[i_level + 1]
                )

            self._encoder.append(encoder_block)

        bottleneck_num = channels[-1]

        # Initializing the bottleneck
        self._bottleneck_res1 = ResidualBlock(
            in_channels=channels[-1],
            out_channels=bottleneck_num,
            dropout=0
        )
        self._bottleneck_attn = Attention(
            num_channels=bottleneck_num
        )
        self._bottleneck_res2 = ResidualBlock(
            in_channels=channels[-1],
            out_channels=bottleneck_num,
            dropout=0
        )

        # Initializing the decoder
        self._decoder = nn.ModuleList()
        for i_level in range(len(channel_multipliers)):

            decoder_block = nn.ModuleDict()
            decoder_block["residuals"] = nn.ModuleList()
            for i_block in range(num_res_blocks + 1):

                block_in = channels[i_level] if i_block != 0 else channels[i_level + 1]
                block_out = channels[i_level]

                block = nn.ModuleDict()
                block["residual"] = ResidualBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    dropout=dropout
                )

                if resolution // 2 ** i_level in attention_resolutions:
                    block["attention"] = Attention(
                        num_channels=block_out
                    )

                decoder_block["residuals"].append(block)

            if i_level != 0:
                decoder_block["upsample"] = Upsample(num_channels=channels[i_level])

            self._decoder.append(decoder_block)

        # End convolution
        self._end = nn.Sequential(
            nn.GroupNorm(
                num_groups=32,
                num_channels=out_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=out_channels,
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
        h = self._bottleneck_res1(h)
        h = self._bottleneck_attn(h)
        h = self._bottleneck_res2(h)

        # Decoder
        for decoder_block in reversed(self._decoder):
            for res_block in decoder_block["residuals"]:
                h = torch.concat([h, hs[-1]], dim=1)
                h = res_block["residual"](h, emb)
                if "attention" in res_block:
                    h = res_block["attention"](h)
            if "upsample" in decoder_block:
                h = decoder_block["upsample"](h)

        return self._end(h)


if __name__ == "__main__":
    sigmas = torch.linspace(0, 80, 16)
    print(f"Shape: {sigmas.shape}")
    emb = NoiseEmbedding(dim=10)
    embeddings = emb(sigmas)
    print(embeddings)


