import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
#from torch.amp import autocast, GradScaler
from torch.cuda.amp import autocast, GradScaler

import subprocess
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm

cfg = dict(
    name="g2",
    steps=24,
    channels=16,
    hidden=128,
    dropout=0.1,
    length=1024
)
latent_dim = cfg["channels"] * cfg["length"]
bit_length = latent_dim

def get_config():
    return cfg  # or return dict(cfg) for a copy


class UrandomRandomWindowDataset(Dataset):
    """
    Dataset that returns random 1024-bit windows from large .bin files without pre-slicing.
    Each sample is:
      - Choose a random .bin file
      - Pick a random byte offset that aligns to 1024-bit (128-byte) boundary
      - Read 128 bytes, unpack bits, return as FloatTensor
    """
    def __init__(self, root_dir, chunk_bits=1024, samples_per_epoch=10000):
        self.files = sorted(Path(root_dir).glob("*.bin"))
        assert len(self.files) > 0, "No .bin files found in root_dir"
        self.chunk_bits = chunk_bits
        self.chunk_bytes = chunk_bits // 8
        self.samples_per_epoch = samples_per_epoch

        # Precompute file sizes
        self.file_sizes = [os.path.getsize(f) for f in self.files]
        self.max_offsets = [fsz - self.chunk_bytes for fsz in self.file_sizes]

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Randomly select file index and offset
        file_idx = np.random.randint(0, len(self.files))
        path = self.files[file_idx]
        max_off = self.max_offsets[file_idx]
        byte_offset = np.random.randint(0, max_off + 1)
        # Align to chunk_bytes boundary
        byte_offset = (byte_offset // self.chunk_bytes) * self.chunk_bytes

        # Read raw bytes
        with open(path, "rb") as f:
            f.seek(byte_offset)
            raw = f.read(self.chunk_bytes)

        # Unpack bits
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:self.chunk_bits]
        return torch.tensor(bits, dtype=torch.float32)


# === NCA Generator ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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



class LinearDiscriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        if x.shape[1] != self.input_size:
            raise ValueError(f" Discriminator input mismatch: got {x.shape[1]}, expected {self.input_size}")
        return self.model(x)




def byte_histogram_uniformity_loss(bitstream):
    """
    Penalizes non-uniform distribution of byte values (0–255)
    from the generated bitstream. Expects binary (0/1) input.
    """
    batch_size, bit_length = bitstream.shape

    # Trim to multiple of 8 bits
    bit_length = (bit_length // 8) * 8
    bitstream = bitstream[:, :bit_length]

    # Reshape to (B, N_bytes, 8)
    bytes_view = bitstream.reshape(batch_size, -1, 8)

    # Convert each 8-bit sequence to a byte (0-255)
    weights = 2 ** torch.arange(7, -1, -1, device=bitstream.device)
    byte_values = (bytes_view * weights).sum(dim=-1)  # Shape: (B, N_bytes)

    # Flatten across batch
    flat_bytes = byte_values.flatten()

    # Histogram of byte values
    # hist = torch.bincount(flat_bytes, minlength=256).float()
    hist = torch.bincount(flat_bytes.long(), minlength=256).float()
    hist = hist / (hist.sum() + 1e-8)  # Normalize

    # Compare with uniform distribution
    uniform = torch.full_like(hist, 1.0 / 256)
    loss = torch.nn.functional.mse_loss(hist, uniform)

    return loss







# === Train Function ===

import subprocess
import re
def entropy_eval_fn1(bin_path):
    result = subprocess.run(['ent', bin_path], stdout=subprocess.PIPE, text=True)
    output = result.stdout

    def extract(pattern, cast=float, default=0.0):
        match = re.search(pattern, output)
        return cast(match.group(1)) if match else default

    entropy = extract(r'Entropy = ([\d.]+)')
    
    compression = extract(r'by (\d+) percent', cast=int) / 100
  
    chi_square = extract(r'Chi square distribution.*?is ([\d.]+)')
   
    mean = extract(r'Arithmetic mean.*?is ([\d.]+)')
   
    pi_error = extract(r'error ([\d.]+) percent')
   
    serial_corr = extract(r'Serial correlation coefficient is ([\-\d.]+)')
  
    return {
        "entropy": entropy,
        "compression": compression,
        "chi_square": chi_square,
        "mean": mean,
        "pi_error": pi_error,
        "serial_corr": serial_corr,
    }






def byte_entropy_loss(bits):
    B, N = bits.shape
    bits = bits[:, :N // 8 * 8]  # Ensure divisible by 8
    byte_chunks = bits.reshape(B, -1, 8)
    byte_vals = (byte_chunks * (2 ** torch.arange(7, -1, -1, device=bits.device))).sum(dim=-1)

    hist = torch.stack([
        torch.histc(b.float(), bins=256, min=0, max=255) for b in byte_vals
    ])
    probs = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
    entropy = - (probs * (probs + 1e-8).log2()).sum(dim=1)
    return -entropy.mean() / 8.0  # Normalize (max byte entropy = 8)


def byte_diversity_loss(bits):
    # Reshape bits into bytes
    bits = bits.view(bits.size(0), -1)
    pad = (8 - bits.size(1) % 8) % 8
    if pad > 0:
        bits = F.pad(bits, (0, pad), value=0)

    byte_chunks = bits.view(bits.size(0), -1, 8)
    byte_vals = (byte_chunks * (2 ** torch.arange(7, -1, -1, device=bits.device))).sum(dim=-1)

    # ✅ Fix: Convert float → tensor
    diversity = torch.stack([
        torch.tensor(len(torch.unique(b)) / 256.0, device=bits.device)
        for b in byte_vals
    ])

    return 1.0 - diversity.mean()  # lower loss = higher diversity

def bit_runlength_loss(bits: torch.Tensor):
    """
    Penalizes long runs of 0s or 1s. Lower loss if bits flip frequently.
    """
    diff = bits[:, 1:] != bits[:, :-1]
    run_lengths = diff.sum(dim=1).float()
    max_flips = bits.size(1) - 1
    return 1.0 - (run_lengths / max_flips).mean()  # ideal = high number of flips

def serial_correlation_loss(bits: torch.Tensor):
    """
    Penalizes correlation between adjacent bits.
    Ideal = 0 (random transitions)
    """
    shifted = torch.roll(bits, shifts=1, dims=1)
    corr = ((bits - bits.mean(dim=1, keepdim=True)) *
            (shifted - shifted.mean(dim=1, keepdim=True))).mean(dim=1)
    return corr.abs().mean()




def train_model(
    generator,
    discriminator,
    loader,
    epochs=1000,
    device='cuda',
    pretrain_epochs=100,
    val_interval=100,
    use_amp=True,
    save_path="/content/drive/MyDrive/CSPRNG/models/best_generator.pt",
    entropy_eval_fn=entropy_eval_fn1
):
    generator.to(device)
    discriminator.to(device)
    latent_dim = generator.channels * generator.length

    opt_G = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    scaler_G = GradScaler(enabled=use_amp)
    scaler_D = GradScaler(enabled=use_amp)
    loss_log_path = "/content/drive/MyDrive/CSPRNG/logs/generator_loss_log.txt"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(loss_log_path), exist_ok=True)
    # Clear file if it exists (truncate to 0 bytes)
    with open(loss_log_path, "w") as f:
        pass  # opening in "w" mode clears the file
    best_entropy = 0.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def fft_flatness_score(bits):
        fft = torch.fft.fft(bits.float())
        mag = torch.abs(fft)
        flatness = (mag.std() / (mag.mean() + 1e-9))
        return flatness

    def hybrid_loss(bits):
        bits = bits.clamp(1e-6, 1 - 1e-6)  # avoid log(0)
        entropy = -bits * torch.log2(bits) - (1 - bits) * torch.log2(1 - bits)
        entropy = entropy.mean()
        monobit = (bits.mean() - 0.5) ** 2
        fft_score = fft_flatness_score(bits)
        return -entropy + 10.0 * monobit + 10.0 * fft_score

    def generator_step(z,epoch=None, step=None):
        B = z.size(0)
        x = z.view(B, generator.channels, generator.length)
        g_out = generator(x)
        
        # Apply temperature to logits before sigmoid
        temperature = 0.8
        logits = g_out.view(B, -1)
        probs = torch.sigmoid(logits / temperature)

        # Adversarial loss
        adv_loss = -discriminator(probs).mean()

        # Core randomness losses
        eps = 1e-8

        # Binarize probs to get actual bits
        bits = (probs > 0.5).to(torch.float32)

        # Compute bit mean
        bit_mean = bits.mean()

        # Compute bit entropy using mean
        entropy_score = -(
            bit_mean * torch.log2(bit_mean + eps) +
            (1 - bit_mean) * torch.log2(1 - bit_mean + eps)
        )
        monobit_score = torch.abs(probs.mean() - 0.5)
        byte_hist_score = byte_histogram_uniformity_loss(probs)
        byte_entropy = byte_entropy_loss(probs)
        byte_div = byte_diversity_loss(probs)
        runlength_score = bit_runlength_loss(probs)
        serial_corr = serial_correlation_loss(probs)

        # FFT flatness
        signals = probs * 2 - 1
        fft_vals = torch.fft.fft(signals, dim=1)
        power = fft_vals.real ** 2 + fft_vals.imag ** 2
        power = power[:, 1:]
        geom = torch.exp(torch.mean(torch.log(power + eps), dim=1))
        arith = torch.mean(power, dim=1) + eps
        fft_score = (1.0 - geom / arith).mean()

        

        # Weighted Loss Terms
        lambda_adv = 0.0
        lambda_entropy = 100.0 * (2048 / latent_dim)
        lambda_monobit = 40.0 * (2048 / latent_dim)
        lambda_fft = 10.0
        lambda_hist = 5.0
        lambda_byte_entropy = 10.0
        lambda_div = 15.0
        lambda_runlength = 10.0
        lambda_serial_corr = 10.0


        total_loss = (
            -lambda_adv * adv_loss +
            -lambda_entropy * entropy_score +
            lambda_monobit * monobit_score +
            lambda_fft * fft_score +
            lambda_hist * byte_hist_score +
            lambda_byte_entropy * byte_entropy +
            lambda_div * byte_div +
            lambda_runlength * runlength_score +
            lambda_serial_corr * serial_corr
        )
        log_line = (
            f"Generator Loss Breakdown:\n"
            f"Epoch {epoch}" + (f", Step {step}" if step is not None else "") + "\n"
            f"Entropy         : {entropy_score.item():.6f}\n"
            f"Monobit         : {monobit_score.item():.6f}\n"
            f"FFT Flatness    : {fft_score.item():.6f}\n"
            f"Byte Histogram  : {byte_hist_score.item():.6f}\n"
            f"Byte Diversity  : {byte_div}/256\n"
            f"Total Gen Loss  : {total_loss.item():.6f}"  
        )
        with open(loss_log_path, "a") as f:
            f.write(log_line + "\n")
        return total_loss, probs, byte_hist_score.item()



      

    def discriminator_step(real_bits, fake_bits, lambda_gp=10.0, noise_std=0.01):
        # Add Gaussian noise for regularization
        real_bits = real_bits + noise_std * torch.randn_like(real_bits)
        fake_bits = fake_bits + noise_std * torch.randn_like(fake_bits)

        real_bits = real_bits.detach()
        fake_bits = fake_bits.detach()

        d_real = discriminator(real_bits)
        d_fake = discriminator(fake_bits)
        loss = d_fake.mean() - d_real.mean()

        # Gradient penalty
        alpha = torch.rand(real_bits.size(0), 1, device=real_bits.device)
        interp = (alpha * real_bits + (1 - alpha) * fake_bits).requires_grad_(True)
        d_interp = discriminator(interp)
        grads = torch.autograd.grad(
            d_interp, interp, torch.ones_like(d_interp),
            create_graph=True
        )[0]
        penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return loss + lambda_gp * penalty
    entropy_history = []
    patience = 3
    no_improve_counter = 0

    for epoch in range(epochs):
        z = torch.randn(64, latent_dim, device=device)
        opt_G.zero_grad()
        with autocast(device, enabled=use_amp):
            loss_G, _, hist_loss = generator_step(z, epoch, step=0)
            #print(f"📘 Epoch {epoch}: Loss_G = {loss_G.item():.4f} | ByteHist Loss = {hist_loss:.6f}")

        if loss_G is not None and torch.isfinite(loss_G):
            scaler_G.scale(loss_G).backward()
            try:
                scaler_G.step(opt_G)
                scaler_G.update()
            except AssertionError:
                print(" AMP Warning: Generator step skipped (no inf checks).")

        # === Limit number of D updates per epoch for speed
        max_batches = 3 # 5 if epoch < 300 else 10  # You can tune this rule
        for i, real_batch in enumerate(loader):
            real_batch = real_batch.to(device)
            z_d = torch.randn(real_batch.size(0), latent_dim, device=device)
            _, fake_bits, _ = generator_step(z_d, epoch, step=i)

            opt_D.zero_grad()
            with autocast(device, enabled=use_amp):
                loss_D = discriminator_step(real_batch, fake_bits)

            if loss_D is not None and torch.isfinite(loss_D):
                scaler_D.scale(loss_D).backward()
                try:
                    scaler_D.step(opt_D)
                    scaler_D.update()
                except AssertionError:
                    print(" AMP Warning: Discriminator step skipped (no inf checks).")

            if i >= max_batches - 1:
                break

        if epoch >= pretrain_epochs and epoch % val_interval == 0 and entropy_eval_fn:
            generator.eval()
            with torch.no_grad():
                z_val = torch.randn(1024, latent_dim, device=device)
                g_out = generator(z_val.view(1024, generator.channels, generator.length))
                probs = torch.sigmoid(g_out.view(1024, -1))
                bits = (probs > 0.5).float()

                # === Save raw binary for ENT ===
                path = f"/content/val_epoch_{epoch}.bin"
                bits_bin = bits.cpu().numpy().astype(np.uint8).flatten()
                with open(path, "wb") as f:
                    f.write(np.packbits(bits_bin).tobytes())

                # === Bit-level diagnostics
                bitstream = bits_bin
                bit_mean = bitstream.mean()
                
               
                bit_entropy = -(
                    probs * torch.log2(probs + 1e-8) +
                    (1 - probs) * torch.log2(1 - probs + 1e-8)
                ).mean().item()
                byte_chunks = bitstream[:len(bitstream) // 8 * 8].reshape(-1, 8)
                byte_vals = np.packbits(byte_chunks.astype(np.uint8), axis=1).flatten()
                byte_hist, _ = np.histogram(byte_vals, bins=256, range=(0, 256), density=True)
                byte_std = byte_hist.std()

                print(f"\n Bit-Level Diagnostics @ Epoch {epoch}")
                print(f"    Bit Mean     : {bit_mean:.5f}")
                print(f"    Bit Entropy  : {bit_entropy:.5f}")
                print(f"    Byte StdDev  : {byte_std:.5f}")
                print(f"    Sample Bytes : {byte_vals[:10].tolist()}")
                bit_counts = (bits == 1).sum().item(), (bits == 0).sum().item()
                print(f"    Bit Histogram  : 1s = {bit_counts[0]}, 0s = {bit_counts[1]}")
                # === Run ENT tool
                stats = entropy_eval_fn(path)
                if isinstance(stats, tuple):
                    ent = stats[0]
                    compression = stats[1]
                    print(f"🧪 ENT Val@{epoch} [TUPLE]: Entropy={ent:.4f}, Compression={compression*100:.2f}%")
                else:
                    ent = stats["entropy"]
                    compression = stats["compression"]
                    print(f"🧪 ENT Val@{epoch}:")
                    print(f"    Entropy        : {ent:.4f} bits/byte")
                    print(f"    Compression    : {compression*100:.2f}%")
                    print(f"    Chi-square     : {stats['chi_square']:.2f}")
                    print(f"    Mean           : {stats['mean']:.2f}")
                    print(f"    Serial Corr.   : {stats['serial_corr']:.5f}")
                    print(f"    Pi Error       : {stats['pi_error']:.2f}%")

                # === Save best only if entropy improves AND bit_mean is good
                if ent > best_entropy and abs(bit_mean - 0.5) < (0.02 if latent_dim >= 8192 else 0.01):
                    best_entropy = ent
                    torch.save(generator.state_dict(), save_path)
                    print(f" New best generator saved @ Epoch {epoch} | Entropy: {ent:.4f}")
                    no_improve_counter = 0  # reset
                else:
                    no_improve_counter += 1
                    print(f"  No improvement in entropy for {no_improve_counter} val steps.")
                    if no_improve_counter >= patience:
                        print(f" Early stopping at epoch {epoch} due to entropy drop.")
                        break

        if epoch % 20 == 0:
            print(f" Epoch {epoch}: Loss_G = {loss_G.item():.4f} | ByteHist Loss = {hist_loss:.6f}")

        torch.cuda.empty_cache()

    print(" Training complete. Best entropy:", best_entropy)
    print(" Best model saved at:", save_path)




# === TorchScript Export ===
def export_torchscript(generator, path="/content/drive/MyDrive/CSPRNG/models/best_generator_g1.pt"):
    generator.eval()
    example_input = torch.randn(1, channels, length)
    traced = torch.jit.trace(generator.cpu(), example_input)
    torch.jit.save(traced, path)
    print(" TorchScript model saved to:", path)
