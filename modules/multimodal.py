import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalEncoder(nn.Module):
    """
    Unified encoder for text, image, and audio features.
    This remains, as it's about perception, not execution.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.text_proj = nn.Linear(embed_dim, embed_dim)
        self.image_proj = nn.Linear(768, embed_dim) 
        self.audio_proj = nn.Linear(128, embed_dim) 
        
        self.modality_embeddings = nn.Parameter(torch.randn(3, embed_dim))
        
    def forward(self, text_feats=None, image_feats=None, audio_feats=None):
        tokens = []
        
        if text_feats is not None:
            t = self.text_proj(text_feats) + self.modality_embeddings[0]
            tokens.append(t)
            
        if image_feats is not None:
            i = self.image_proj(image_feats) + self.modality_embeddings[1]
            tokens.append(i)
            
        if audio_feats is not None:
            a = self.audio_proj(audio_feats) + self.modality_embeddings[2]
            tokens.append(a)
            
        if not tokens:
            return torch.zeros(1, 0, self.embed_dim, device=self.modality_embeddings.device)
            
        return torch.cat(tokens, dim=1)

class ToolCallHead(nn.Module):
    """
    Generates structured tool calls based on the model's hidden state.
    Adheres to the Nivaya Tool Interface Standard (NTIS).
    """
    def __init__(self, embed_dim, max_tools=20, max_actions_per_tool=5, max_param_tokens=50):
        super().__init__()
        self.max_tools = max_tools
        self.max_actions_per_tool = max_actions_per_tool
        self.max_param_tokens = max_param_tokens

        # Core NTIS fields
        self.tool_selector = nn.Linear(embed_dim, max_tools)
        self.action_selector = nn.Linear(embed_dim, max_actions_per_tool)
        self.param_generator = nn.Linear(embed_dim, max_param_tokens) 
        
        # Constraints prediction (Cost, Timeout, Approval)
        self.constraint_predictor = nn.Linear(embed_dim, 3) 

    def forward(self, hidden_state):
        last_state = hidden_state[:, -1, :]
        
        tool_logits = self.tool_selector(last_state)
        action_logits = self.action_selector(last_state)
        param_tokens = self.param_generator(last_state)
        constraints_raw = torch.sigmoid(self.constraint_predictor(last_state))
        
        return {
            "tool_logits": tool_logits,
            "action_logits": action_logits,
            "param_tokens": param_tokens,
            "constraints_raw": constraints_raw
        }

class Nivaya3Model(nn.Module):
    """
    The refactored Nivaya 3.0 Model Architecture: a pure cognitive decision engine.
    """
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, num_experts=8):
        super().__init__()
        from .attention import NivayaBlock
        from .moe import NivayaMoEBlock
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.mm_encoder = MultiModalEncoder(embed_dim)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                self.layers.append(NivayaBlock(embed_dim, num_heads, is_attention=(i%4==0)))
            else:
                self.layers.append(NivayaMoEBlock(embed_dim, num_heads, num_experts=num_experts))
                
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size) # Still useful for internal reasoning/text generation
        self.tool_call_head = ToolCallHead(embed_dim)
        
    def forward(self, input_ids=None, image_feats=None, audio_feats=None, mask=None):
        from .moe import NivayaMoEBlock
        from .attention import NivayaBlock
        
        tokens = []
        if input_ids is not None:
            tokens.append(self.embedding(input_ids))
            
        mm_tokens = self.mm_encoder(text_feats=None, image_feats=image_feats, audio_feats=audio_feats)
        if mm_tokens.shape[1] > 0:
            if input_ids is not None and mm_tokens.shape[0] != input_ids.shape[0]:
                mm_tokens = mm_tokens.expand(input_ids.shape[0], -1, -1)
            tokens.append(mm_tokens)
            
        if not tokens:
            # Handle case where no input is provided, or provide a default start token
            # For a cognitive engine, this might imply waiting for input or generating a default action
            # For now, return None or raise error if no input is acceptable.
            raise ValueError("Nivaya 3.0 requires at least one form of input (text, image, or audio).")
            
        x = torch.cat(tokens, dim=1)
        
        for layer in self.layers:
            if isinstance(layer, NivayaMoEBlock) or (isinstance(layer, NivayaBlock) and "FlashAttention" in str(type(layer.fn))):
                 x = layer(x, mask=mask)
            else:
                 x = layer(x)
                 
        x = self.norm(x)
        
        logits = self.lm_head(x)
        tool_call_output = self.tool_call_head(x)
        
        return {
            "logits": logits,
            "tool_call_output": tool_call_output,
            "last_hidden_state": x
        }
