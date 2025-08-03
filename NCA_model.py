class NCAGenerator(nn.Module):
    def __init__(self, steps=16, channels=16, hidden=128, dropout=0.1, length=128):
        super().__init__()
        self.steps = steps
        self.channels = channels
        self.length = length

        self.conv1 = nn.Conv1d(channels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.conv3 = nn.Conv1d(hidden, channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm([channels, length])  # Apply after each step

    def step_fn(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv3(out)

        return x + out  # residual update

    def forward(self, x):
        for _ in range(self.steps):
            x = checkpoint(self.step_fn, x, use_reentrant=False)

            # 🧪 Add tiny noise for regularization (helps with diversity)
            if self.training:
                x = x + torch.randn_like(x) * 0.01

            # 🧹 LayerNorm for stability
            x = self.norm(x)

        return self.dropout(x)  # final output logits


