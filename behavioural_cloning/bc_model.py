# bc_model.py
import torch
import torch.nn as nn


# ---------- text ----------
def build_vocab(instructions):
    words = sorted({w for text in instructions for w in text.split()})
    return {w: i + 1 for i, w in enumerate(words)}   # 0 is kept for padding


def encode(text, vocab, max_len=8):
    ids = [vocab[w] for w in text.split()]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids)


# ---------- network ----------
class ImageEncoder(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, d_model), nn.ReLU(),
        )

    def forward(self, image):
        x = self.net(image)        # (B, d_model, 8, 8)
        x = x.flatten(2)           # (B, d_model, 64)
        return x.transpose(1, 2)   # (B, 64, d_model)


class InputEmbedder(nn.Module):
    def __init__(self, vocab_size, d_model=128, max_text=8, n_steps=7, n_cells=64):
        super().__init__()
        self.image_encoder = ImageEncoder(d_model)
        self.image_pos = nn.Parameter(torch.randn(1, n_cells, d_model) * 0.02)
        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.word_pos = nn.Parameter(torch.randn(1, max_text, d_model) * 0.02)
        self.step_emb = nn.Embedding(n_steps, d_model)
        self.finger_proj = nn.Linear(1, d_model)

    def forward(self, image, tokens, finger, step):
        img = self.image_encoder(image) + self.image_pos
        txt = self.word_emb(tokens) + self.word_pos
        stp = self.step_emb(step).unsqueeze(1)
        fin = self.finger_proj(finger.unsqueeze(1) * 25).unsqueeze(1)
        return torch.cat([img, txt, stp, fin], dim=1)   # (B, 74, d)


class PolicyBody(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=3, n_heads=4, chunk=4):
        super().__init__()
        self.chunk = chunk
        self.embedder = InputEmbedder(vocab_size, d_model)
        self.queries = nn.Parameter(torch.randn(1, chunk, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=256,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, image, tokens, finger, step):
        x = self.embedder(image, tokens, finger, step)
        q = self.queries.expand(x.shape[0], -1, -1)
        x = torch.cat([x, q], dim=1)
        x = self.encoder(x)
        return x[:, -self.chunk:]


class BCPolicy(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.body = PolicyBody(vocab_size, d_model)
        self.q_head = nn.Linear(d_model, 7)
        self.gripper_head = nn.Linear(d_model, 1)

    def forward(self, image, tokens, finger, step):
        h = self.body(image, tokens, finger, step)
        q = self.q_head(h)                           # (B, 4, 7), normalized
        g = self.gripper_head(h).squeeze(-1)         # (B, 4), logits
        return q, g


# ---------- loading a trained checkpoint ----------
def load_policy(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    vocab = ckpt["vocab"]
    model = BCPolicy(vocab_size=len(vocab) + 1).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["q_mean"].to(device), ckpt["q_std"].to(device), vocab

@torch.no_grad()
def predict(model, q_mean, q_std, vocab, image, instruction, finger, step, device="cpu"):
    img = torch.from_numpy(image).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    tokens = encode(instruction, vocab).unsqueeze(0).to(device)
    finger_t = torch.tensor([finger], dtype=torch.float32, device=device)
    step_t = torch.tensor([step], device=device)

    q_norm, g_logit = model(img, tokens, finger_t, step_t)

    q = (q_norm[0] * q_std + q_mean).cpu().numpy()      # (4, 7) joint angles in radians
    gripper_open = (g_logit[0] > 0).cpu().numpy()       # (4,) True = open
    return q, gripper_open