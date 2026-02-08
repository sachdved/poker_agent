import torch
import numpy
import typing

class StreetPositionalEncoding(torch.nn.Module):
    """
    Generates an embedding for the street we are on in a poker hand.
    Note there are 4 streets in texas hold 'em: pre flop, flop, turn, river.
    In addition, there is an end of street token, end of (observed) action token, hand complete token, and pad token.
    """
    def __init__(
        self,
        num_streets: int = 4,
        embedding_dim: int = 2048,
        max_seq_len: int = 128,
        device : str = 'cpu'
    ):
        super().__init__()
        self.num_streets = num_streets
        self.embedding_dim = embedding_dim
        self.street_embedder = torch.nn.Embedding(num_streets+3, embedding_dim)
        self.max_seq_len = max_seq_len
        self.end_of_street_token = num_streets
        self.end_of_hand_token = num_streets + 1
        self.pad_token = num_streets + 2
        self.device = device
        self.to(self.device)

    def forward(
        self,
        street_idxs: torch.Tensor
    ):
        """
        Returns embeddings for the street of the action in the action sequence.
        """
        B, L = street_idxs.shape
    
        idxs = torch.full(
            (B, self.max_seq_len),
            fill_value=self.pad_token,
            dtype=torch.long,
            device=self.device
        )
    
        idxs[:, :L] = street_idxs
    
        return idxs, self.street_embedder(idxs)

class TablePositionalEncoding(torch.nn.Module):
    """
    Generates an embedding for the position of the player acting in a poker hand.
    Note there are 9 positions in texas hold 'em: small blind, big blind, utg, utg+1, 
    middle position, lojack, hijack, cutoff, button. These will be represented as integers
    in terms of distance from the small blind. In addition, there is an end of action token and a pad
    token.
    """
    def __init__(
        self,
        num_players: int = 8,
        embedding_dim: int = 2048,
        max_seq_len: int = 128,
        device : str = "cpu"
    ):
        super().__init__()
        self.num_players = num_players
        self.embedding_dim = embedding_dim
        self.player_embedder = torch.nn.Embedding(num_players+2, embedding_dim)
        self.max_seq_len = max_seq_len
        self.new_round_token = num_players
        self.pad_token = num_players + 1
        self.device = device
        self.to(self.device)

    def forward(
        self,
        player_idxs: torch.Tensor  # (B, L)
    ):
        """
        Batched operation of padding sequences.
        """
        B, L = player_idxs.shape
    
        idxs = torch.full(
            (B, self.max_seq_len),
            fill_value=self.pad_token,
            dtype=torch.long,
            device=self.device
        )
    
        idxs[:, :L] = player_idxs
    
        return idxs, self.player_embedder(idxs)

class ActionEncoding(torch.nn.Module):
    """
    Encodes the possible actions that could occur that are taken with by the player with the particular position 
    and on the particular street. The possible actions are put in small blind,
    put in big blind, fold, check, call, bet, and then 10 raise actions, be inactive because folded, or be inactive
    because all in. The raise sizes are ranging from min-raise to all in logarithmically. There are 6 special tokens:
    inactive because folded, inactive because all in, end of street token, end of hand token, and a pad token.
    """
    def __init__(
        self,
        embedding_dim : int = 2048,
        max_seq_len : int = 128,
        device : str = "cpu"
    ):
        super().__init__()
        self.num_actions = 22

        # Indices representing what action means what.
        self.action_fold = 2
        self.action_check = 3
        self.action_call = 4
        self.action_min_bet = 5
        self.action_bet_raise_min = 6  # Min raise when facing bet, or small bet when no bet
        self.action_bet_raise_all_in = 16  # All-in (works as bet or raise)
        
        # Blind actions (not real player decisions)
        self.action_post_sb = 0
        self.action_post_bb = 1
        
        # Inactive player markers
        self.action_inactive_folded = 17
        self.action_inactive_allin = 18
        
        # Game flow control actions
        self.action_street_complete = 19  # Betting round complete, move to next street
        self.action_hand_complete = 20    # Hand is over (showdown or all folded)
        self.pad_token = 21
        
        
        self.action_embedder = torch.nn.Embedding(self.num_actions, embedding_dim)
        self.max_seq_len = max_seq_len
        self.end_of_action_token = self.num_actions - 2
        self.pad_token = self.num_actions - 1
        self.device = device
        self.to(self.device)

    def forward(
        self,
        action_idxs: torch.Tensor,  # (B, L_max)
    ):
        B, L = action_idxs.shape
    
        idxs = torch.full(
            (B, self.max_seq_len),
            fill_value=self.pad_token,
            dtype=torch.long,
            device=self.device
        )
    
        idxs[:, :L] = action_idxs

        return idxs, self.action_embedder(idxs)

class PotSizeSequenceEmbedder(torch.nn.Module):
    """
    A class just to make sure that the pot size sequence is padded to the right length. 
    Does no transformations other than to pad (and maybe make sure everything
    lives on the right device).
    """
    def __init__(
        self,
        max_seq_len : int = 128,
        pad_value : float = -1.,
        device : str = 'cpu'
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.device = device
        self.to(self.device)

    def forward(
        self,
        pot_size_sequence
    ):
        padded_pot_size_sequence = torch.full(
            (pot_size_sequence.shape[0], self.max_seq_len),
            fill_value = self.pad_value,
            dtype = torch.float,
            device = self.device
        )

        padded_pot_size_sequence[:, :pot_size_sequence.shape[1]] = pot_size_sequence
        return padded_pot_size_sequence

class PokerSequenceEmbedder(torch.nn.Module):
    """
    Projects per-timestep poker features into a shared latent space
    suitable as input to a Transformer.
    """

    def __init__(
        self,
        street_input_dimension: int,
        table_position_input_dimension: int,
        action_input_dimension: int,
        latent_dimensions: typing.List[int],
        device: str = 'cpu',
    ):
        super().__init__()

        self.device = device

        def make_mlp(input_dim: int) -> torch.nn.Sequential:
            dims = [input_dim] + latent_dimensions
            layers = []
            for i in range(len(dims) - 1):
                layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
                layers.append(torch.nn.LayerNorm(dims[i + 1]))
                if i < len(dims) - 2:  # no ReLU on final layer
                    layers.append(torch.nn.ReLU())
            return torch.nn.Sequential(*layers)

        self.street_MLP = make_mlp(street_input_dimension)
        self.table_position_MLP = make_mlp(table_position_input_dimension)
        self.action_MLP = make_mlp(action_input_dimension)
        self.pot_MLP = make_mlp(1)

        self.to(device)

    def forward(self, model_inputs: dict) -> dict:
        """
        Parameters:
            model_inputs : dict with keys
                street_embedding: Tensor (B, T, D_street).
                table_position_embedding: Tensor (B, T, D_position).
                action_embedding: Tensor (B, T, D_action).
                pot_size_sequence: Tensor (B, T, 1).

        Returns:
            model_inputs : dict
                Same dict with additional key:
                sequence_embedding: Tensor (B, T, D_latent).
        """

        street = model_inputs["street_embedding"]
        position = model_inputs["table_position_embedding"]
        action = model_inputs["action_embedding"]
        pot = model_inputs["pot_size_sequence"]

        street_latent = self.street_MLP(street)
        position_latent = self.table_position_MLP(position)
        action_latent = self.action_MLP(action)
        pot_latent = self.pot_MLP(pot)

        return street_latent + position_latent + action_latent + pot_latent