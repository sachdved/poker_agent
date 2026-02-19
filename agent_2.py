import torch
import transformers
import ml_modules


class PokerAgent(torch.nn.Module):
    """
    Full poker agent.

    Structure:
        learnable embedders → frozen LLM → learnable policy
    """

    def __init__(
        self,
        cards: ml_modules.Cards,
        street_embedder: torch.nn.Module,
        table_position_embedder: torch.nn.Module,
        action_embedder: torch.nn.Module,
        pot_size_embedder: torch.nn.Module,
        poker_sequence_embedder: torch.nn.Module,
        llm: transformers.AutoModelForCausalLM,
        policy_model: torch.nn.Module,
        device: str = "cuda",
        llm_train: bool = False,
        llm_layer_to_read: int = 27,
    ):
        super().__init__()

        self.device = device
        self.llm_train = llm_train
        self.llm_layer_to_read = llm_layer_to_read

        self.cards = cards
        self.street_embedder = street_embedder
        self.table_position_embedder = table_position_embedder
        self.action_embedder = action_embedder
        self.pot_size_embedder = pot_size_embedder
        self.poker_sequence_embedder = poker_sequence_embedder
        self.llm = llm
        self.policy_model = policy_model

        # --------------------------------------------------
        # Freeze LLM if requested
        # --------------------------------------------------
        if not llm_train:
            for p in self.llm.parameters():
                p.requires_grad = False

            # Enable checkpointing inside the transformer
            self.llm.gradient_checkpointing_enable()

            # Turn off caching (required for checkpointing)
            if hasattr(self.llm.config, "use_cache"):
                self.llm.config.use_cache = False

        self.to(device)

    # ======================================================
    # Forward
    # ======================================================
    def forward(
        self,
        self_position_idx: torch.Tensor,
        card_idxs: torch.Tensor,
        street_idxs: torch.Tensor,
        table_position_idxs: torch.Tensor,
        action_idxs: torch.Tensor,
        pot_size_sequence: torch.Tensor,
        active_players: torch.Tensor,
        stack_size: torch.Tensor,
    ):

        # ---------------------------
        # Embeddings (learnable)
        # ---------------------------
        card_embeddings = self.cards(card_idxs[:, 0, :], card_idxs[:, 1, :])

        street_idxs_out, street_embs = self.street_embedder(street_idxs)
        table_pos_idxs_out, table_pos_embs = self.table_position_embedder(
            table_position_idxs
        )
        action_idxs_out, action_embs = self.action_embedder(action_idxs)

        padded_pot_size_sequence = self.pot_size_embedder(pot_size_sequence)

        model_inputs = (
            {
                "street_idxs": street_idxs_out,
                "street_embedding": street_embs,
                "table_position_idxs": table_pos_idxs_out,
                "table_position_embedding": table_pos_embs,
                "action_idxs": action_idxs_out,
                "action_embedding": action_embs,
                "pot_size_sequence": padded_pot_size_sequence.unsqueeze(2),
                "active_players": active_players,
                "stack_size": stack_size,
                "card_embeddings": card_embeddings,
            }
        )

        attention_mask = (
            (model_inputs["pot_size_sequence"] != -1)
            .squeeze(-1)
            .to(self.device)
        )

        # ---------------------------
        # Sequence embedding
        # ---------------------------
        inputs_embeds = (
            self.poker_sequence_embedder(model_inputs)
            .to(device=self.device, dtype=torch.bfloat16)
        )

        # ---------------------------
        # LLM forward (checkpointed internally)
        # ---------------------------
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden state from desired layer
        hidden = outputs.hidden_states[self.llm_layer_to_read + 1]

        # Take last valid token
        last_token_idx = attention_mask.sum(dim=1) - 1
        llm_state = hidden[
            torch.arange(hidden.shape[0], device=self.device),
            last_token_idx,
            :
        ]

        # ---------------------------
        # Policy head (learnable)
        # ---------------------------
        probits, value_pred = self.policy_model(
            self_position_idx,
            active_players,
            stack_size,
            card_embeddings,
            llm_state,
        )

        return {
            "probits": probits,
            "value_pred": value_pred,
            "llm_state": llm_state,
        }
