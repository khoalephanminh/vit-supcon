class PromptedVisionTransformer(nn.Module):
    def __init__(self, 
                 image_size: int,
                 image_patch_size: int,
                 frames: int,
                 frame_patch_size : int,
                 num_layers: int,
                 num_heads: int,
                 hidden_dim: int,
                 mlp_dim: int,
                 dropout: 0.0,
                 emb_dropout: 0.0,
                 num_classes = 5,
                 channels = 3,
                 dim_head = 64,
                 freeze_vit = True,
                 pool = 'cls',
                 pretrain_path = None,
                 num_prompts = 8,
                 prompt_dropout = 0.0,
                 ):
        super().__init__()
        self.image_size = image_size
        self.num_layers = num_layers
        self.image_patch_size = image_patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.emb_dropout = emb_dropout
        self.dropout = dropout
        self.prompt_dropout = prompt_dropout
        self.num_classes = num_classes

        # PROMPT GENERATOR
        self.prompt_generator = PromptNet(hidden_dim = hidden_dim)
        
        self.vision_transformer = VisionTransformer(
                                     image_size=image_size, 
                                     image_patch_size=image_patch_size, 
                                     frames=frames, 
                                     frame_patch_size=frame_patch_size, 
                                     num_classes=num_classes, 
                                     dim=hidden_dim, 
                                     depth=num_layers, 
                                     heads=num_heads, 
                                     mlp_dim=mlp_dim, 
                                     pool = pool, 
                                     channels = channels, 
                                     dim_head = dim_head, 
                                     dropout = dropout, 
                                     emb_dropout = emb_dropout, 
                                     pretrain_path=pretrain_path)
        self.freeze_vit = freeze_vit
        
        self.init_head_weights()
        
        if self.freeze_vit:
            for k, p in self.vision_transformer.named_parameters():
                if ("transformer" in k or "cls_token" in k or "conv_proj" in k or "pos_embedding" in k):
                    p.requires_grad = False

            for k, p in self.prompt_generator.named_parameters():
                if "head" in k or "prompt_mlp" in k:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        # Tunable Prompt
        self.tunable_prompt_tokens = nn.Parameter(torch.randn(num_prompts, hidden_dim))

        # Init Cross Attn
        self.prompt_norm = nn.LayerNorm(hidden_dim)
        self.scale_down_gate = nn.Linear(hidden_dim, 64)
        self.scale = 64 ** -0.5 
        self.prompt_dropout = nn.Dropout(prompt_dropout)
        self.scale_up_gate = nn.Linear(64, hidden_dim)

    def split_embedding(self, embedding, prompt_dim1):
        """Split the embedding into cls, prompt tokens, and image tokens."""
        cls = embedding[:, :1, :] 
        p = embedding[:, 1:1+prompt_dim1, :] # Prompt tokens (bs, 200, dim)
        x = embedding[:, 1+prompt_dim1:, :] # Image tokens (bs, 1000, dim)
        return cls, p, x

    def prompt_fusing(self, static_prompt):
        """Fuse static and tunable prompts."""
        b, _, _ = static_prompt.shape  # bs, prompt_dim1, hidden_dim
        tunable_prompt = self.prompt_dropout(self.tunable_prompt_tokens).expand(b, -1, -1)  # bs, num_prompts, hidden_dim
        return torch.cat((static_prompt, tunable_prompt), dim=1)  # bs, prompt_dim1 + num_prompts, hidden_dim

    def merge_embedding(self, cls, p, x):
        """Merge cls, prompt, and image tokens."""
        return torch.cat((cls, p, x), dim=1)
        
    
    def _adapter(self, embedding, prompt_dim1):
        """Perform the adapter logic."""
        cls, p, x = self.split_embedding(embedding, prompt_dim1)
        p = self.prompt_fusing(p)  # Fuse static and tunable prompts (bs, 208, hidden dim)
        expanded_prompt_dim1 = p.shape[1]  # Update prompt_dim to include added prompts
        embedding = self.merge_embedding(cls, p, x)  # Merge back (bs, 1209, hidden_dim)

        # Down scale
        embedding_down = self.prompt_norm(embedding)
        embedding_down = self.scale_down_gate(embedding_down) # bs, 1209, 64
        cls_down, K, Q = self.split_embedding(embedding_down, expanded_prompt_dim1) 
        """
        Q (bs, 1000, 64)
        K (bs, 208, 64)
        V (bs, 208, 64)
        """
        V = K.clone()
        attn_scores = (torch.bmm(Q, K.transpose(1,2)) * self.scale) # (bs, 1000, 208)
        attn_weights = F.softmax(attn_scores, dim=-1)
        weighted_prompts = torch.bmm(attn_weights, V) # (bs, 1000, 64)
        enhanced_image_tokens = Q + weighted_prompts 

        # Up scale
        final_embedding = self.merge_embedding(cls_down, K,enhanced_image_tokens) # (bs, 1209, 64)
        final_embedding = F.gelu(final_embedding) 
        final_embedding = self.scale_up_gate(final_embedding)  # (bs, 1209, 768)
        return embedding + final_embedding

    def init_head_weights(self):
        nn.init.xavier_uniform_(self.vision_transformer.mlp_head.weight)
        nn.init.zeros_(self.vision_transformer.mlp_head.bias)
        print("Initialize head weight successfully!")
    
    def train(self, mode=True):
        if mode:
            # Ensure ViT encoder stays in eval mode if frozen
            super().train(mode)
            if self.freeze_vit:
                self.prompt_generator.eval()
                self.prompt_generator.encoder.projection_head.train()
                self.prompt_generator.prompt_mlp.train()
                
                self.vision_transformer.transformer.eval()
                self.vision_transformer.conv_proj.eval()
                self.vision_transformer.dropout.eval()
                self.vision_transformer.mlp_head.train()
                
                self.prompt_norm.train()
                self.scale_down_gate.train()
                self.prompt_dropout.train()
                self.scale_up_gate.train()
                
        else:
            # Set all submodules to evaluation mode
            for module in self.children():
                module.eval()

    def _generate_prompt(self, p: torch.Tensor) -> torch.Tensor:
        return self.prompt_generator(p)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.vision_transformer.conv_proj(x)
    
    def forward_deep_prompt(self, x, p):
        """Forward logic for deep prompt tuning."""
        prompt_dim1 = p.shape[1] # 200
        for i in range(self.num_layers):
            if i == 0:
                x = torch.cat((x[:,:1,:], p, x[:,1:,:]), dim=1) # Combine prompt embeddings with image embeddings (after cls tokens before patch embeddings) at the first Transformer layer
            else:
                x = torch.cat((x[:,:1,:], p, x[:,(1 + p.shape[1]):,:]), dim=1) # after cls tokens & prompt embeddings at the current layers (to avoid overiding prompt embedding in previous, help keep info from current layers) before patch embeddings
            
            attn, ff = self.vision_transformer.transformer.layers[i]
            x = attn(x) + x
            if i % 3 == 0:
                x = self._adapter(x, prompt_dim1)  # Keep prompt_dim consistent
            x = ff(x) + x
        x = self.vision_transformer.transformer.norm(x)    
        x = x.mean(dim=1) if self.vision_transformer.pool == 'mean' else x[:, 0]
        x = self.vision_transformer.to_latent(x)
        return self.vision_transformer.mlp_head(x)


    def forward(self, x: torch.Tensor, p: torch.Tensor): 
        p = self._generate_prompt(p) # prompt embedding (B, 200, 768)
        x = self._process_input(x) 
        x = x.flatten(2).transpose(1,2) # [B, N, C]
        b, n, _ = x.shape

        cls_tokens = repeat(self.vision_transformer.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.vision_transformer.pos_embedding[:, :(n + 1)]
        x = self.vision_transformer.dropout(x)
        
        return self.forward_deep_prompt(x, p)