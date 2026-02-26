import torch
import numpy
import typing

class Cards(torch.nn.Module):
    """
    Encodes the cards in a 52 card deck, including a mask card, to represent the hidden cards.
    """
    def __init__(
        self,
        number_of_ranks : int = 14,
        number_of_suits : int = 5,
        embedding_dim : int = 1024,
        device : str = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.rank_embedder = torch.nn.Embedding(number_of_ranks, embedding_dim)
        self.suit_embedder = torch.nn.Embedding(number_of_suits, embedding_dim)
        self.to(self.device)

    def forward(
        self,
        rank_idxs,
        suit_idxs
    ):
        return torch.cat(
            [self.rank_embedder(rank_idxs), self.suit_embedder(suit_idxs)], 
            dim = -1
        )

class SelfPositionEmbedder(torch.nn.Module):
    """
    Encodes the position of self in each scenario.
    """
    def __init__(
        self,
        number_of_positions : int,
        embedding_dim : int = 2048,
        device : str = "cpu"
    ):
        super().__init__()
        self.device = device
        self.number_of_positions = number_of_positions
        self.position_embedder = torch.nn.Embedding(number_of_positions, embedding_dim)
        self.to(self.device)

    def forward(
        self,
        position_idx
    ):
        position_idx = position_idx.long()
        return self.position_embedder(position_idx)
        

class PolicyModel(torch.nn.Module):
    """
    Building the model that outputs the logits over the action space for best response.
    """
    def __init__(
        self,
        num_players : int,
        self_position_embedder : torch.nn.Module,
        active_players_hidden_dims : typing.List[int],
        stack_size_hidden_dims : typing.List[int],
        card_embeddings_hidden_dims : typing.List[int],
        final_output_hidden_dims : typing.List[int],
        value_output_hidden_dims : typing.List[int],
        num_actions : int = 21,  # fold, check/call, bet/raise small, bet/raise large
        card_embedding_dim : int = 2048,
        dropout_rate : float = 0.1,
        card_aggregation : str = "deepset",  # "deepset" or "sum"
        device : str = "cpu"
    ):
        """
        Initializes the policy model. It takes in the specifications
        for each model and builds an MLP. Note that the final_output_hidden_dims
        will take in an input that is assumed to be 4 * 2048 (2048 dimensions for each
        embedding - active players, stack size, card embeddings, and then the state embedding
        from the LLM).
        """
        super(PolicyModel, self).__init__()

        self.self_position_embedder = self_position_embedder
        
        self.num_players = num_players
        self.num_actions = num_actions
        self.card_embedding_dim = card_embedding_dim
        self.card_aggregation = card_aggregation
        self.device = device
        
        # Active players embedding network
        # Input: num_players (binary indicator of active players)
        active_players_layers = []
        prev_dim = num_players
        for hidden_dim in active_players_hidden_dims:
            active_players_layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            active_players_layers.append(torch.nn.ReLU())
            active_players_layers.append(torch.nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        active_players_layers.append(torch.nn.Linear(prev_dim, 2048))
        self.active_players_net = torch.nn.Sequential(*active_players_layers).to(self.device)
        
        # Stack size embedding network
        # Input: num_players (stack sizes for each player)
        stack_size_layers = []
        prev_dim = num_players
        for hidden_dim in stack_size_hidden_dims:
            stack_size_layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            stack_size_layers.append(torch.nn.ReLU())
            stack_size_layers.append(torch.nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        stack_size_layers.append(torch.nn.Linear(prev_dim, 2048))
        self.stack_size_net = torch.nn.Sequential(*stack_size_layers).to(self.device)
        
        # Card embeddings network (DeepSet architecture)
        # Input: [batch_size, num_cards, card_embedding_dim]
        # Process each card individually, then aggregate
        if card_aggregation == "deepset":
            # Per-card processing network (phi function in DeepSet)
            card_phi_layers = []
            prev_dim = card_embedding_dim
            for hidden_dim in card_embeddings_hidden_dims:
                card_phi_layers.append(torch.nn.Linear(prev_dim, hidden_dim))
                card_phi_layers.append(torch.nn.ReLU())
                card_phi_layers.append(torch.nn.Dropout(dropout_rate))
                prev_dim = hidden_dim
            card_phi_layers.append(torch.nn.Linear(prev_dim, 2048))
            self.card_phi_net = torch.nn.Sequential(*card_phi_layers).to(self.device)
            
            # Post-aggregation network (rho function in DeepSet)
            # Takes summed representation and processes it further
            card_rho_layers = []
            prev_dim = 2048
            for hidden_dim in card_embeddings_hidden_dims:
                card_rho_layers.append(torch.nn.Linear(prev_dim, hidden_dim))
                card_rho_layers.append(torch.nn.ReLU())
                card_rho_layers.append(torch.nn.Dropout(dropout_rate))
                prev_dim = hidden_dim
            card_rho_layers.append(torch.nn.Linear(prev_dim, 2048))
            self.card_rho_net = torch.nn.Sequential(*card_rho_layers).to(self.device)
        else:  # Simple summation
            # Just process the summed card embeddings directly
            card_sum_layers = []
            prev_dim = card_embedding_dim
            for hidden_dim in card_embeddings_hidden_dims:
                card_sum_layers.append(torch.nn.Linear(prev_dim, hidden_dim))
                card_sum_layers.append(torch.nn.ReLU())
                card_sum_layers.append(torch.nn.Dropout(dropout_rate))
                prev_dim = hidden_dim
            card_sum_layers.append(torch.nn.Linear(prev_dim, 2048))
            self.card_sum_net = torch.nn.Sequential(*card_sum_layers).to(self.device)
        
        # Final output network
        # Input: 4 * 2048 (concatenated embeddings)
        final_output_layers = []
        prev_dim = 5 * 2048
        for hidden_dim in final_output_hidden_dims:
            final_output_layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            final_output_layers.append(torch.nn.ReLU())
            final_output_layers.append(torch.nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        final_output_layers.append(torch.nn.Linear(prev_dim, num_actions))
        self.final_output_net = torch.nn.Sequential(*final_output_layers).to(self.device)

        final_output_layers = []
        prev_dim = 5 * 2048
        for hidden_dim in value_output_hidden_dims:
            final_output_layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            final_output_layers.append(torch.nn.ReLU())
            final_output_layers.append(torch.nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        final_output_layers.append(torch.nn.Linear(prev_dim, 1))
        self.value_output_net = torch.nn.Sequential(*final_output_layers).to(self.device)
        
    def forward(
        self, 
        self_position : torch.Tensor,
        active_players : torch.Tensor,  # Shape: (batch_size, num_players)
        stack_sizes : torch.Tensor,     # Shape: (batch_size, num_players)
        card_embeddings : torch.Tensor,   # Shape: (batch_size, num_cards, card_embedding_dim)
        llm_state_embedding : torch.Tensor  # Shape: (batch_size, 2048)
    ) -> torch.Tensor:
        """
        Forward pass through the policy network.
        
        Args:
            active_players: Binary indicators of which players are active
            stack_sizes: Stack sizes for each player
            card_embeddings: Card embeddings with shape (batch_size, num_cards, card_embedding_dim)
            llm_state_embedding: Pre-computed embedding from LLM (2048-dim)
            
        Returns:
            logits: Action logits with shape (batch_size, num_actions)
        """
        # Generate embeddings from each component
        self_position_embed = self.self_position_embedder(self_position)
        active_emb = self.active_players_net(active_players)  # (batch_size, 2048)
        stack_emb = self.stack_size_net(stack_sizes)          # (batch_size, 2048)
        
        # Process card embeddings using DeepSet or summation
        if self.card_aggregation == "deepset":
            # Apply phi to each card independently
            batch_size, num_cards, card_dim = card_embeddings.shape
            # Reshape to process all cards in batch
            cards_flat = card_embeddings.view(batch_size * num_cards, card_dim)
            phi_out = self.card_phi_net(cards_flat)  # (batch_size * num_cards, 2048)
            phi_out = phi_out.view(batch_size, num_cards, 2048)
            
            # Aggregate (sum) over cards dimension
            aggregated = phi_out.sum(dim=1)  # (batch_size, 2048)

            
            # Apply rho to aggregated representation
            card_emb = self.card_rho_net(aggregated)  # (batch_size, 2048)

        else:  # Simple summation
            # Sum over cards dimension first
            summed_cards = card_embeddings.sum(dim=1)  # (batch_size, card_embedding_dim)
            card_emb = self.card_sum_net(summed_cards)  # (batch_size, 2048)
        
        # Concatenate all embeddings
        

        combined = torch.cat([
            self_position_embed,
            active_emb, 
            stack_emb, 
            card_emb, 
            llm_state_embedding
        ], dim=1)  # (batch_size, 10240)
        
        # Generate action logits
        logits = self.final_output_net(combined)  # (batch_size, num_actions)
        value_pred = self.value_output_net(combined)
        
        return logits, value_pred
    
    def get_action_probs(
        self,
        active_players : torch.Tensor,
        stack_sizes : torch.Tensor,
        card_embeddings : torch.Tensor,
        llm_state_embedding : torch.Tensor
    ) -> torch.Tensor:
        """
        Returns action probabilities using softmax.
        
        Returns:
            probs: Action probabilities with shape (batch_size, num_actions)
        """
        logits = self.forward(active_players, stack_sizes, card_embeddings, llm_state_embedding)
        return torch.nn.functional.softmax(logits, dim=-1)
    
    def sample_action(
        self,
        active_players : torch.Tensor,
        stack_sizes : torch.Tensor,
        card_embeddings : torch.Tensor,
        llm_state_embedding : torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples an action from the policy distribution.
        
        Returns:
            actions: Sampled actions with shape (batch_size,)
            log_probs: Log probabilities of sampled actions with shape (batch_size,)
        """
        probs = self.get_action_probs(active_players, stack_sizes, card_embeddings, llm_state_embedding)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs