"""
01 - MLP binary classifier, end to end.

Upgrades over the minimal version (each one is a common interview follow-up):
  * train / val split           -> you can't judge generalization on training data
  * mini-batches via DataLoader -> SGD noise, memory scaling
  * BCEWithLogitsLoss           -> numerically stable (log-sum-exp trick inside)
  * AdamW + cosine schedule     -> decoupled weight decay, smooth LR decay
  * proper train()/eval() modes -> matters for Dropout / BatchNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.datasets import make_moons

SEED = 42
torch.manual_seed(SEED)
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# --- 1. Data ------------------------------------------------
X_np, y_np = make_moons(n_samples=1000, noise=0.2, random_state=SEED)
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

ds = TensorDataset(X, y)
train_ds, val_ds = random_split(ds, [800, 200], generator=torch.Generator().manual_seed(SEED))
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, shuffle=False)


# --- 2. Model ------------------------------------------------
model = nn.Sequential(
    nn.Linear(2, 16), nn.ReLU(),
    nn.Linear(16, 16), nn.ReLU(),
    nn.Linear(16, 1), nn.Flatten(0),
).to(device)

# weight initlialization
def init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        if m is model[-2]:
            nn.init.xavier_normal_(m.weight)
        else:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)

model.apply(init_weights)

# Total # of trainable parameters
params = sum(p.numel() for p in model.parameters())
print(f"The total number of trainable parameters: {params}.")

# Setting up the optimizer and LR scheduler
EPOCHS = 50
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=2e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_dl))

# --- 3. Training Loop ------------------------------------------------
def train_one_epoch():
    model.train()
    total = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        train_logits = model(xb)
        train_loss = F.binary_cross_entropy_with_logits(train_logits, yb)

        optimizer.zero_grad(set_to_none=False)
        train_loss.backward()
        optimizer.step()
        scheduler.step()
        total += train_loss.item() * len(xb)

    return total / len(train_dl.dataset)

# --- 4. Validation Loop ------------------------------------------------
@torch.inference_mode()
def evaluate(dl):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for xb, yb in dl:
       xb, yb = xb.to(device), yb.to(device)
       val_logits = model(xb)
       loss_sum += F.binary_cross_entropy_with_logits(val_logits, yb, reduction="sum")

       correct += ((val_logits > 0).float() == yb).sum().item() # logits > 0 <=> p > 0.5
       n += len(xb) 

    return loss_sum / n, correct / n

# --- 5. Loop ------------------------------------------------
for epoch in range(EPOCHS):
    tr_loss = train_one_epoch()
    if epoch % 10 == 0 or epoch == 1:
        val_loss, val_acc = evaluate(val_dl)
        print(f"epoch {epoch:3d} | train {tr_loss:.4f} | val {val_loss:.4f} | val acc {val_acc*100:.1f}%")

# Save the model after training
torch.save(model.state_dict(), 'model_mlp.pt' )

