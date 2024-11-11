import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import ARCDataset
import os
import json
from sklearn.metrics import mean_squared_error

class TransformerEncoder(nn.Module):
    def __init__(self, grid_size=30, patch_size=2, embed_dim=128, num_heads=8, depth=6, ff_dim=128, dropout_prob=0.2):
        super(TransformerEncoder, self).__init__()
        
        # Compute number of patches based on grid size and patch size
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.num_patches = (grid_size // patch_size) ** 2  # 15x15 patches for 30x30 grid with 2x2 patches
        self.patch_dim = patch_size * patch_size  # Each patch is 2x2 (for 1 channel)
        
        # Linear layer for embedding patches into embedding space
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        
        # Positional Embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

        # CLS token (classification token for global representation)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Stack of Transformer Encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout_prob)
            for _ in range(depth)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Reshape input grid into patches
        batch_size = x.shape[0]
        x = self._reshape_to_patches(x)  # (batch_size, num_patches, patch_dim)
        
        # Apply patch embedding
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        
        # Concatenate CLS token and add positional embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, embed_dim)
        x += self.pos_embedding

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Final normalization
        x = self.norm(x)

        # Return CLS token (global representation)
        return x[:, 0]  # (batch_size, embed_dim)

    def _reshape_to_patches(self, x):
        """
        Reshape a 30x30 grid into 2x2 patches.
        """
        batch_size, height, width = x.shape
        assert height == self.grid_size and width == self.grid_size, "Grid size should be 30x30"
        patch_size = self.patch_size

        # Reshape to patches
        x = x.reshape(batch_size, height // patch_size, patch_size, width // patch_size, patch_size)
        x = x.permute(0, 1, 3, 2, 4)  # (batch_size, num_patches_h, num_patches_w, patch_size, patch_size)
        x = x.reshape(batch_size, -1, self.patch_dim)  # (batch_size, num_patches, patch_dim)
        
        return x

    def evaluate(self, val_loader, device):
        """
        Evaluate the model on validation data using MSE loss
        """
        self.eval()  # Set model to evaluation mode
        total_mse = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = self(inputs)  # Using self instead of model since we're in the class
                
                # Convert to numpy for MSE calculation
                outputs_np = outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                mse = mean_squared_error(targets_np.flatten(), outputs_np.flatten())
                total_mse += mse
                num_batches += 1
        
        avg_mse = total_mse / num_batches
        return avg_mse

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_prob):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_prob)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Self-attention mechanism
        attn_output, _ = self.mha(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))  # Add & norm
        
        # Feedforward network
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))  # Add & norm
        
        return x

class DotProductLoss(nn.Module):
    def __init__(self):
        super(DotProductLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, embedding_A, embedding_B, num_transformations):
        # Compute dot product between embeddings
        dot_product = torch.sum(embedding_A * embedding_B, dim=1)  # (batch_size, )

        # Target dot product: 1 / num_transformations
        target_dot_product = 1.0 / num_transformations

        # Compute MSE loss between dot product and target value
        loss = self.mse_loss(dot_product, target_dot_product)

        return loss

encoder_model = TransformerEncoder()
loss_fn = DotProductLoss()
optimizer = torch.optim.AdamW(encoder_model.parameters(), lr=1e-2)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
dataset_dir = os.path.join("data", "training")

# Split train dataset 
train_dataset = ARCDataset("data", split="train", include_mutations=True)
valid_dataset = ARCDataset("data", split="valid", include_mutations=True)
print("Imported training dataset:", len(train_dataset.data))
print("Imported valid dataset of size: ", len(valid_dataset.data))
pair_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

input()
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)
encoder_model.to(device)

# Load and evaluate saved model
if os.path.exists('./dist_model'):
    saved_model = torch.load('./dist_model', map_location=device)
    saved_model.eval()
    val_mse = saved_model.evaluate(val_loader, device)
    print(f"Validation MSE for saved model: {val_mse:.4f}")

for epoch in range(num_epochs):
    encoder_model.train()
    running_loss = 0.0
    
    for grid_A, grid_B, num_trans in pair_loader:
        grid_A, grid_B, num_trans = grid_A.to(device), grid_B.to(device), num_trans.float().to(device)
        
        # Forward pass through Transformer Encoder
        embedding_A = encoder_model(grid_A)
        embedding_B = encoder_model(grid_B)

        # Compute loss
        loss = loss_fn(embedding_A, embedding_B, num_trans)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    avg_loss = running_loss / len(pair_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

torch.save(encoder_model, "./dist_model")

val_mse = encoder_model.evaluate(val_loader, device)
print(f"Validation MSE: {val_mse:.4f}")
