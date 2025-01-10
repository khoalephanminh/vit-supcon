class PromptNet(nn.Module):
    def __init__ (self, hidden_dim = 768):
        super().__init__()
        self.encoder = prompt_model
        self.prompt_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)  # (bs, 768, 8, 5, 5)
        b, c, d, h, w = z.shape
        z = z.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, c) # (bs, 200, 768)
        return self.prompt_mlp(z)