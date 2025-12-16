import torch
from torch import nn


class SpacingFiLM(nn.Module):
    def __init__(self, feature_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, feature_channels * 2)  # gamma + beta
        )

    def forward(self, x, spacing):
        film = self.mlp(spacing)  # B, 2C
        gamma, beta = film.chunk(2, dim=1)

        # reshape to B,C,1,1,1
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]
        return x * gamma + beta


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and a skip connection"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Handle channel mismatch for residual connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class Down(nn.Module):
    """Downsampling with strided convolution"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """Upsampling with transposed convolution"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.PReLU()

    def forward(self, x):
        if hasattr(self, 'conv'):  # For trilinear case
            x = self.up(x)
            x = self.conv(x)
        else:
            x = self.up(x)
        return self.relu(self.bn(x))


class VNet(nn.Module):
    def __init__(self, model="s", n_channels=1, n_classes=4, trilinear=True, spacing_aware=False):
        super().__init__()

        self.spacing_aware = spacing_aware
        self.base_c = {"xs": 4, "s": 8, "m": 16, "l": 32}[model]

        C = self.base_c

        # ---------------- Encoder ----------------
        self.enc0 = ResidualBlock(n_channels, C)
        self.down1 = Down(C, C * 2)
        self.enc1 = ResidualBlock(C * 2, C * 2)

        self.down2 = Down(C * 2, C * 4)
        self.enc2 = ResidualBlock(C * 4, C * 4)

        self.down3 = Down(C * 4, C * 8)
        self.enc3 = ResidualBlock(C * 8, C * 8)

        self.down4 = Down(C * 8, C * 16)
        self.enc4 = ResidualBlock(C * 16, C * 16)

        # -------------- Spacing FiLM ----------------
        if spacing_aware:
            self.film0 = SpacingFiLM(C)
            self.film1 = SpacingFiLM(C * 2)
            self.film2 = SpacingFiLM(C * 4)
            self.film3 = SpacingFiLM(C * 8)
            self.film4 = SpacingFiLM(C * 16)

        # ---------------- Decoder ----------------
        self.up1 = Up(C * 16, C * 8, trilinear)
        self.dec1 = ResidualBlock(C * 16, C * 8)

        self.up2 = Up(C * 8, C * 4, trilinear)
        self.dec2 = ResidualBlock(C * 8, C * 4)

        self.up3 = Up(C * 4, C * 2, trilinear)
        self.dec3 = ResidualBlock(C * 4, C * 2)

        self.up4 = Up(C * 2, C, trilinear)
        self.dec4 = ResidualBlock(C * 2, C)

        # ---------------- Output ----------------
        self.outc = nn.Conv3d(C, n_classes, kernel_size=1)

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    def forward(self, x, spacing=None):

        # ----- encoder -----
        x0 = self.enc0(x)
        if self.spacing_aware:
            x0 = self.film0(x0, spacing)

        x1 = self.enc1(self.down1(x0))
        if self.spacing_aware:
            x1 = self.film1(x1, spacing)

        x2 = self.enc2(self.down2(x1))
        if self.spacing_aware:
            x2 = self.film2(x2, spacing)

        x3 = self.enc3(self.down3(x2))
        if self.spacing_aware:
            x3 = self.film3(x3, spacing)

        x4 = self.enc4(self.down4(x3))
        if self.spacing_aware:
            x4 = self.film4(x4, spacing)

        # ----- decoder -----
        x = self.up1(x4)
        x = self.dec1(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, x1], dim=1))

        x = self.up4(x)
        x = self.dec4(torch.cat([x, x0], dim=1))

        return self.outc(x)
