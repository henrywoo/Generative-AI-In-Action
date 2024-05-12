import torch
import torch.nn as nn
from torch_scatter import scatter
from transformers.models.t5.modeling_t5 import T5LayerNorm
from torch.utils.checkpoint import checkpoint


# Create expert class (can be any neural network architecture)
class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# Create MoE layer
class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, k=2):
        super().__init__()
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
        self.k = k
        self.norm = T5LayerNorm(d_model)

    def forward(self, x):
        gate_output = self.gate(x)
        top_expert_indices = torch.topk(gate_output, self.k, dim=-1).indices

        expert_outputs = [checkpoint(expert, x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        x = scatter(
            expert_outputs[torch.arange(x.size(0)).unsqueeze(1), top_expert_indices],
            top_expert_indices.view(-1),
            dim=0,
            reduce="mean",
        ).view(x.shape)

        return self.norm(x)


if __name__ == '__main__':
    from hiq.vis import print_model
    moe = MoELayer(d_model=512, d_ff=8)
    print_model(moe)