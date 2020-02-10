import torch
import torch.nn as nn
import torch.nn.functional as F


class EmulatedDMoE(nn.Module):
    def __init__(self, in_features, num_experts, num_active, update_every_inputs, update_every_steps, 
                 Expert, Optimizer):
        """
        A local mixture of experts module that emulates the behavior of DMoE from Learning@home
        The emulation concerns two main aspects:
          * Individual experts are updated not by the trainer but automatically after accumulating enough gradients
          * Forward/backward pass takes 
          
        
        Warning: experts should NOT be optimized manually! Use get_non_expert_parameters(model) for optimizer

        :param in_features: input dimension
        :param num_experts: total number of experts
        :param num_active: only *this many* experts with highest score participate in a computation
        :param update_every_inputs: automatically triggers an update for an expert after it was used 
            on *this many* inputs; this counter resets after every update
        :param update_every_steps: automatic update happens at most *this many* steps after expert processed
            its first input; this counter resets after every update
        :param Expert: callable(dim)-> nn.Module that receives and returns vectors of this dimension
        :param Optimizer: callable(Sequence[nn.Parameter]) -> torch.optim.Optimizer
        """
        super().__init__()
        self.gating_pre_normalize = nn.LayerNorm(in_features)
        self.expert_keys = nn.Parameter(torch.randn(in_features, num_experts))
        
        self.experts = nn.ModuleList([Expert(in_features) for _ in range(num_experts)])
        self.expert_optimizers = {expert: Optimizer(expert.parameters()) for expert in self.experts}
        self.register_buffer('expert_inputs_since_update', torch.zeros(num_experts, dtype=torch.int64))
        self.register_buffer('expert_steps_since_first_input', torch.zeros(num_experts, dtype=torch.int64))
        
        self.num_active = num_active
        self.update_every_inputs = update_every_inputs
        self.update_every_steps = update_every_steps

    def forward(self, input):
        assert len(input.shape) == 2
        batch_size = len(input)
        if self.training:
            self.maybe_update_experts()
        
        gating_logits = self.gating_pre_normalize(input) @ F.normalize(self.expert_keys, dim=-1)
        chosen_ids = torch.argsort(gating_logits, dim=-1, descending=True)[..., :self.num_active]
        
        outputs = []
        for i in range(batch_size):
            chosen_experts = [self.experts[chosen_id] for chosen_id in chosen_ids[i]]

            weights = F.softmax(gating_logits[i][chosen_ids[i]], dim=-1)
            expert_outputs = torch.stack([expert(input[i]) for expert in chosen_experts], dim=-1)
            output = expert_outputs @ weights
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        
        
        # update expert usage counts
        if self.training:
            self.expert_inputs_since_update.scatter_add_(
                0, chosen_ids.reshape(-1), 
                torch.ones_like(chosen_ids, device=self.expert_inputs_since_update.device).view(-1))
            self.expert_steps_since_first_input += (self.expert_inputs_since_update > 0).to(torch.int64)
        return outputs
    
    def maybe_update_experts(self):
        for i, expert in enumerate(self.experts):
            if self.expert_inputs_since_update[i] >= self.update_every_inputs or \
               self.expert_steps_since_first_input[i] >= self.update_every_steps:
                self.expert_optimizers[expert].step()
                self.expert_optimizers[expert].zero_grad()
                self.expert_inputs_since_update[i] = 0
                self.expert_steps_since_first_input[i] = 0
                
def get_non_expert_params(model):
    expert_params = set(param for module in model.modules() if isinstance(module, EmulatedDMoE)
                    for param in module.parameters())
    return [param for param in model.parameters() if param not in expert_params]

