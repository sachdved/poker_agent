import torch
import transformers
import ml_modules

class PokerAgent(torch.nn.Module):
    """
    Full poker agent. Contains information about the cards,
    the players, the board, the sequence embedding, and the 
    probability prediction model.
    """
    def __init__(
        self,
        cards : ml_modules.Cards,
        street_embedder : torch.nn.Module,
        table_position_embedder : torch.nn.Module,
        action_embedder : torch.nn.Module,
        pot_size_embedder : torch.nn.Module,
        poker_sequence_embedder : torch.nn.Module,
        llm : transformers.AutoModelForCausalLM,
        policy_model : torch.nn.Module,
        device : str = "cpu",
        llm_train : bool = False
    ):
        super().__init__()
        self.device = device
        self.llm_train = llm_train

        self.cards = cards
        self.street_embedder = street_embedder
        self.table_position_embedder = table_position_embedder
        self.action_embedder = action_embedder
        self.pot_size_embedder = pot_size_embedder
        self.poker_sequence_embedder = poker_sequence_embedder  # This was missing the definition
        self.llm = llm
        self.policy_model = policy_model
        
        if not llm_train:
            for parameter in self.llm.parameters():
                parameter.requires_grad = False
        
        # Remove the hook from __init__ - it's defined twice and the first one does nothing
       
    def forward(
        self,
        self_position_idx : torch.Tensor,
        card_idxs : torch.Tensor,
        street_idxs : torch.Tensor,
        table_position_idxs : torch.Tensor,
        action_idxs : torch.Tensor,
        pot_size_sequence : torch.Tensor,
        active_players : torch.Tensor,
        stack_size : torch.Tensor,
    ):
        card_embeddings = self.cards(card_idxs[:, 0, :], card_idxs[:, 1, :])
        
        street_idxs_out, street_embs = self.street_embedder(street_idxs)
        street_embedding = {
            'street_idxs': street_idxs_out,
            'street_embedding': street_embs,
        }
        
        table_pos_idxs_out, table_pos_embs = self.table_position_embedder(table_position_idxs)
        table_position_embedding = {
            'table_position_idxs': table_pos_idxs_out,
            'table_position_embedding': table_pos_embs,
        }
        
        action_idxs_out, action_embs = self.action_embedder(action_idxs)
        action_embedding = {
            'action_idxs': action_idxs_out,
            'action_embedding': action_embs,
        }
        padded_pot_size_sequence = self.pot_size_embedder(pot_size_sequence)
        
        model_inputs = (
            street_embedding 
            | 
            table_position_embedding 
            | 
            action_embedding 
            | 
            {'pot_size_sequence' : padded_pot_size_sequence.unsqueeze(2)}
            |
            {
                'active_players' : active_players,
                'stack_size' : stack_size,
                'card_embeddings' : card_embeddings
            }
        )
        model_inputs['attention_mask'] = (model_inputs['pot_size_sequence'] != -1).squeeze(-1).to(self.device)
        
        # Use a dict to capture the activation (mutable object)
        activation_cache = {'activation': None}
        
        def hook(module, input, output):
            activation_cache['activation'] = output
        
        handle = self.llm.model.layers[27].post_attention_layernorm.register_forward_hook(hook)
        
        try:
            inputs_embeds = self.poker_sequence_embedder(model_inputs).to(device="cuda", dtype=torch.bfloat16)
            outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=model_inputs['attention_mask'])
            
        finally:
            handle.remove()
        
        # Get activation from the cache
        activation = activation_cache['activation']
        
        if activation is None:
            raise RuntimeError("Hook did not capture activation - check layer path")
        
        activations_last_action = activation[
            torch.arange(activation.shape[0]), 
            model_inputs['attention_mask'].sum(dim=1) - 1, 
            :
        ]
        
        model_inputs['llm_state'] = activations_last_action
        model_inputs['self_position'] = self_position_idx
        model_inputs['probits'], model_inputs['value_pred'] = self.policy_model(
                model_inputs['self_position'],
                model_inputs['active_players'],
                model_inputs['stack_size'],
                model_inputs['card_embeddings'],
                model_inputs['llm_state']
        )
        return model_inputs