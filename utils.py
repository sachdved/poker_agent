import torch
import numpy as np
import typing

class PokerActionValidator:
    """
    Validates poker actions on the basis of the previous streets of betting history.
    """

    def __init__(
        self,
        num_players : int = 2,
        small_blind : float = 1.,
        big_blind : float = 2.,
        starting_stack_sizes : float = 400.,
    ):
        """
        Initializes the starting state of the game.

        Parameters:
            num_players : how many players are there?
            small_blind : We use unitless representation, so 1.
            big_blind : We use unitless representation, so 2.
            starting_stack_sizes : We assume players start each hand
                with 200 big blinds, so assume 400.
        """
        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.starting_stack_sizes = starting_stack_sizes

        # How many possible valid actions are there? 
        # We now have 21 total actions:
        # 0, 1: small and big blind (forced)
        # 2: fold
        # 3: check
        # 4: call (match existing bet)
        # 5: min bet (initiate betting with minimum size)
        # 6-16: bet/raise sizes (11 logarithmically spaced sizes)
        #       - When no bet exists: these are different BET sizes
        #       - When facing a bet: these are different RAISE sizes
        # 17: folded (inactive)
        # 18: all-in (inactive)
        # 19: street complete (transition to next street)
        # 20: hand complete (end of hand)
        self.num_actions = 21

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

        # Street indices in street_idxs
        self.street_preflop = 0
        self.street_flop = 1
        self.street_turn = 2
        self.street_river = 3
        self.street_transition_token = 4  # Marker for street transition
        self.hand_complete_token = 5      # Marker for hand completion
        self.street_pad_token = 6         # Padding for unobserved actions
        
        # Position padding token
        self.position_pad_token = 99      # Used when position is not applicable (street/hand transitions)

    def get_next_to_act(
        self,
        street_idxs : torch.Tensor,
        table_position_idxs : torch.Tensor,
        action_idxs : torch.Tensor,
        active_players : torch.Tensor,
        all_in_players : typing.Optional[torch.Tensor] = None
    ):
        """
        Identifies which player is next to act for each batch element.

        Parameters: 
            street_idxs : torch.Tensor, [batch_size, seq_len]. Street indicators.
            table_position_idxs : torch.Tensor, [batch_size, seq_len]. Table positions.
            action_idxs : torch.Tensor, [batch_size, seq_len]. Actions taken.
            active_players : torch.Tensor, [batch_size, num_players]. Active player mask.
            all_in_players : torch.Tensor, [batch_size, num_players]. Optional all-in mask.
        
        Returns:
            next_to_act : torch.Tensor, [batch_size]. Position index of next player to act.
                Returns -1 if no player can act (hand complete) or -2 for street transition.
        """
        device = street_idxs.device
        batch_size = street_idxs.shape[0]
        next_to_act = torch.full((batch_size,), -1, dtype=torch.long, device=device)

        for elem in range(batch_size):
            active_players_row = active_players[elem]
            street_row = street_idxs[elem]
            position_row = table_position_idxs[elem]
            action_row = action_idxs[elem]

            # Filter out padding
            valid_mask = street_row < self.street_pad_token
            if not valid_mask.any():
                continue
            
            valid_streets = street_row[valid_mask]
            valid_positions = position_row[valid_mask]
            valid_actions = action_row[valid_mask]
            
            last_idx = len(valid_streets) - 1
            last_street_value = valid_streets[last_idx].item()
            
            # Check for street transition in progress
            if last_idx >= 0 and valid_actions[last_idx].item() == self.action_street_complete:
                
                next_to_act[elem] = torch.where(active_players_row == 1)[0].min()  # Street transition marker
                continue
            
            # Check if on street transition token
            if last_street_value == self.street_transition_token:
                # Find previous street to determine which street we're starting
                for i in range(last_idx - 1, -1, -1):
                    prev_street = valid_streets[i].item()
                    if prev_street < self.street_transition_token:
                        # Starting new street - find first to act post-flop
                        next_position = self._get_first_to_act_postflop(
                            active_players_row, 
                            all_in_players[elem] if all_in_players is not None else None
                        )
                        if next_position is None:
                            next_to_act[elem] = -1  # No one can act
                        else:
                            next_to_act[elem] = next_position
                        break
                continue
            
            # Check if hand complete
            if last_street_value == self.hand_complete_token:
                next_to_act[elem] = -1  # Hand over
                continue
            
            # On an actual street
            relevant_street = last_street_value
            current_street_mask = (street_row == relevant_street) & valid_mask
            position_row_relevant = position_row[current_street_mask]
            action_row_relevant = action_row[current_street_mask]
            
            # Determine next position
            if len(position_row_relevant) == 0:
                # No actions yet on this street
                if relevant_street == self.street_preflop:
                    # Preflop starts with position 0 (SB/BTN in heads-up)
                    next_position = 0
                else:
                    # Post-flop
                    next_position = self._get_first_to_act_postflop(
                        active_players_row,
                        all_in_players[elem] if all_in_players is not None else None
                    )
                    if next_position is None:
                        next_to_act[elem] = -1
                        continue
            else:
                last_position = position_row_relevant[-1].item()
                if last_position == self.position_pad_token:
                    next_position = 0
                else:
                    next_position = int((last_position + 1) % self.num_players)
            
            # Check if street/hand is complete BEFORE checking player status
            if self._is_street_complete(
                action_row_relevant, 
                position_row_relevant,
                active_players_row,
                all_in_players[elem] if all_in_players is not None else None,
                next_position,
                relevant_street
            ):
                if self._is_hand_complete(active_players_row, all_in_players[elem] if all_in_players is not None else None):
                    next_to_act[elem] = -1  # Hand complete
                else:
                    next_to_act[elem] = -2  # Street complete
                continue
            
            # Now check if next player can act
            if active_players_row[next_position] == 0:
                # Player folded - shouldn't normally happen in well-formed sequences
                # but return position anyway with understanding it's inactive
                next_to_act[elem] = next_position
            elif all_in_players is not None and all_in_players[elem][next_position] == 1:
                # Player all-in
                next_to_act[elem] = next_position
            else:
                # Player can act
                next_to_act[elem] = next_position
        
        return next_to_act

    def get_legal_actions_mask(
        self,
        street_idxs : torch.Tensor,
        table_position_idxs : torch.Tensor,
        action_idxs : torch.Tensor,
        pot_size_sequence : torch.Tensor,
        active_players : torch.Tensor,
        player_stacks : typing.Optional[torch.Tensor] = None,
        all_in_players : typing.Optional[torch.Tensor] = None
    ):
        """
        Identifies what the legal action mask should be, based on player status.

        Trying to do this in a batched setting.

        Parameters: 
            street_idxs : torch.Tensor, [batch_size, seq_len]. This indicates to us
                which street we are on. Special values:
                - 0-3: Actual streets (preflop, flop, turn, river)
                - 4: Street transition marker
                - 5: Hand complete marker
                - 6: Padding (unobserved actions)
            table_position_idxs : torch.Tensor, [batch_size, seq_len]. The table position of each 
                player acting, relative to the small blind.
                - 0 to num_players-1: Actual positions
                - position_pad_token (99): Padding for transitions/unobserved
            action_idxs : torch.Tensor, [batch_size, seq_len]. The actions the players have taken
                thus far.
            pot_size_sequence : torch.Tensor, [batch_size, seq_len]. Sequence of pot sizes thus far.
            active_players : torch.Tensor, [batch_size, num_players]. A boolean mask indicating which players remain
                in the hand for each element in the batch (not folded).
            player_stacks : torch.Tensor, [batch_size, num_players]. Optional stack sizes for each player.
            all_in_players : torch.Tensor, [batch_size, num_players]. Optional boolean mask indicating which 
                players are all-in.
        
        Returns:
            legal_actions : torch.Tensor, [batch_size, num_actions]. Boolean mask of legal actions.
        """
        # Get device so we materialize everything on the right machine.
        device = street_idxs.device
        batch_size = street_idxs.shape[0]
        legal_actions = torch.zeros((batch_size, self.num_actions), dtype=torch.bool, device=device)

        # Loop over batches.
        for elem in range(batch_size):

            # Get this batch example.
            active_players_row = active_players[elem]
            street_row = street_idxs[elem]
            position_row = table_position_idxs[elem]
            action_row = action_idxs[elem]
            pot_row = pot_size_sequence[elem]

            # Find the current street by looking backwards from the end
            # Skip padding tokens and find the last actual street or transition marker
            valid_mask = street_row < self.street_pad_token  # Filter out padding
            if not valid_mask.any():
                # All padding - no valid data
                continue
            
            valid_streets = street_row[valid_mask]
            valid_positions = position_row[valid_mask]
            valid_actions = action_row[valid_mask]
            valid_pots = pot_row[valid_mask]
            
            last_idx = len(valid_streets) - 1
            last_street_value = valid_streets[last_idx].item()
            
            # Check if last action was street_complete - means we're transitioning
            if last_idx >= 0 and valid_actions[last_idx].item() == self.action_street_complete:
                # Last action was street_complete
                # Check if the corresponding street_idxs entry is a transition token
                # If street_idxs[same_index] == transition_token, the transition is already recorded
                
                if valid_streets[last_idx].item() == self.street_transition_token:
                    # Transition token is at the same index as street_complete action
                    # This means transition is already recorded, we're starting new street
                    # Fall through to the transition handling logic below
                    pass
                else:
                    # No transition token yet, need to signal environment to add it
                    # Look at what street we were on
                    current_street_idx = last_idx
                    while current_street_idx >= 0 and valid_streets[current_street_idx].item() >= self.street_transition_token:
                        current_street_idx -= 1
                    
                    if current_street_idx >= 0:
                        completed_street = valid_streets[current_street_idx].item()
                        next_street = completed_street + 1
                        
                        # Check if we've reached end of hand (after river)
                        if next_street > self.street_river:
                            legal_actions[elem][self.action_hand_complete] = True
                            continue
                        
                        # Otherwise, we're transitioning to next street
                        # Signal to add street_transition_token
                        legal_actions[elem][self.action_street_complete] = True
                        continue
            
            # Check if we just completed a transition (transition token is last valid entry)
            # If so, we need to figure out what street we're NOW on
            if last_street_value == self.street_transition_token:
                # We're starting a new street - need to determine which one
                # Look backwards to find the last actual street before this transition
                for i in range(last_idx - 1, -1, -1):
                    prev_street = valid_streets[i].item()
                    if prev_street < self.street_transition_token:
                        # Found the previous street, next street is prev + 1
                        relevant_street = prev_street + 1
                        # No actions yet on this new street
                        street_row_relevant = torch.tensor([], dtype=street_row.dtype, device=device)
                        position_row_relevant = torch.tensor([], dtype=position_row.dtype, device=device)
                        action_row_relevant = torch.tensor([], dtype=action_row.dtype, device=device)
                        pot_row_relevant = torch.tensor([], dtype=pot_row.dtype, device=device)
                        
                        # Determine first player to act on this street
                        # Post-flop: first active player after button (position 0 is button/SB in heads-up)
                        # In multi-way: SB is position 0, so start from position 0
                        next_position_to_act = self._get_first_to_act_postflop(active_players_row, all_in_players[elem] if all_in_players is not None else None)
                        
                        if next_position_to_act is None:
                            # No one can act (all folded or all-in)
                            legal_actions[elem][self.action_hand_complete] = True
                            break
                        
                        # Check if this player is active
                        if active_players_row[next_position_to_act] == 0:
                            legal_actions[elem][self.action_inactive_folded] = True
                            break
                        
                        if all_in_players is not None and all_in_players[elem][next_position_to_act] == 1:
                            legal_actions[elem][self.action_inactive_allin] = True
                            break
                        
                        # First to act post-flop: can check or bet any size
                        legal_actions[elem][self.action_check] = True
                        legal_actions[elem][self.action_min_bet:self.action_bet_raise_all_in + 1] = True
                        break
                continue
            
            if last_street_value == self.hand_complete_token:
                # Hand is over
                legal_actions[elem][self.action_hand_complete] = True
                continue
            
            # We're on an actual street (0-3), find the current street
            relevant_street = last_street_value
            
            # Subset everything to current street's actions
            current_street_mask = (street_row == relevant_street) & valid_mask
            street_row_relevant = street_row[current_street_mask]
            position_row_relevant = position_row[current_street_mask]
            action_row_relevant = action_row[current_street_mask]
            pot_row_relevant = pot_row[current_street_mask]

            # Identify next player to act
            if len(position_row_relevant) == 0:
                # No actions yet on this street, first player is position 0
                next_position_to_act = 0
            else:
                last_position = position_row_relevant[-1].item()
                # Skip position_pad_token entries
                if last_position == self.position_pad_token:
                    # This shouldn't happen in well-formed data on actual streets
                    next_position_to_act = 0
                else:
                    next_position_to_act = int((last_position + 1) % self.num_players)
        
            # IMPORTANT: Check if street/hand is complete BEFORE checking player status
            # This handles cases where everyone folds and we shouldn't ask the last player to act
            if self._is_street_complete(
                action_row_relevant, 
                position_row_relevant,
                active_players_row,
                all_in_players[elem] if all_in_players is not None else None,
                next_position_to_act,
                relevant_street
            ):
                # Determine if this is end of street or end of hand
                if self._is_hand_complete(active_players_row, all_in_players[elem] if all_in_players is not None else None):
                    legal_actions[elem][self.action_hand_complete] = True
                else:
                    legal_actions[elem][self.action_street_complete] = True
                continue
            
            # Now check if next player is active (not folded)
            if active_players_row[next_position_to_act] == 0:
                legal_actions[elem][self.action_inactive_folded] = True
                continue
            
            # Check if next player is all-in
            if all_in_players is not None and all_in_players[elem][next_position_to_act] == 1:
                legal_actions[elem][self.action_inactive_allin] = True
                continue
            
            # Player is active - determine legal actions
            # Filter out inactive player actions (both folded and all-in) AND blind postings
            action_row_relevant_active = action_row_relevant[
                (action_row_relevant != self.action_inactive_folded) & 
                (action_row_relevant != self.action_inactive_allin) &
                (action_row_relevant != self.action_post_sb) &
                (action_row_relevant != self.action_post_bb) &
                (action_row_relevant != self.action_street_complete) &
                (action_row_relevant != self.action_hand_complete)
            ]
            
            # If no active actions yet this street, special handling
            if len(action_row_relevant_active) == 0:
                # First to act this street (no real actions yet, only blinds/inactive markers)
                if relevant_street == self.street_preflop:
                    # Preflop first action after blinds: facing the big blind
                    # Must fold, call BB, or raise
                    legal_actions[elem][self.action_fold] = True
                    legal_actions[elem][self.action_call] = True
                    legal_actions[elem][self.action_bet_raise_min:self.action_bet_raise_all_in + 1] = True
                else:
                    # Post-flop first action: can check or bet (min bet + larger bets 6-16)
                    legal_actions[elem][self.action_check] = True
                    legal_actions[elem][self.action_min_bet:self.action_bet_raise_all_in + 1] = True
                continue
            
            # Special case for preflop: check if this is SB acting after others have called/raised
            # On preflop, SB needs special handling because they posted a blind but haven't made a "real" decision yet
            if relevant_street == self.street_preflop and next_position_to_act == 0:
                # Check if SB has made a real decision (not just posted blind)
                # Filter position_row_relevant to only include real actions (not blinds)
                real_action_mask = (
                    (action_row_relevant != self.action_post_sb) &
                    (action_row_relevant != self.action_post_bb) &
                    (action_row_relevant != self.action_inactive_folded) &
                    (action_row_relevant != self.action_inactive_allin)
                )
                
                if real_action_mask.sum() > 0:
                    positions_with_real_actions = position_row_relevant[real_action_mask]
                    sb_made_real_decision = (positions_with_real_actions == 0).any().item()
                else:
                    sb_made_real_decision = False
                
                if not sb_made_real_decision:
                    # SB hasn't made a real decision yet, only posted blind
                    # Check if there's been any raising
                    has_raise = (action_row_relevant_active >= self.action_bet_raise_min).any().item()
                    
                    # SB is always facing at least the BB, so they can fold, call, or raise
                    legal_actions[elem][self.action_fold] = True
                    legal_actions[elem][self.action_call] = True
                    
                    # Can raise if no one has gone all-in
                    has_all_in = (action_row_relevant_active == self.action_bet_raise_all_in).any().item()
                    if not has_all_in:
                        if player_stacks is not None:
                            player_stack = player_stacks[elem][next_position_to_act].item()
                            if player_stack > self.big_blind:
                                legal_actions[elem][self.action_bet_raise_min:self.action_bet_raise_all_in + 1] = True
                        else:
                            legal_actions[elem][self.action_bet_raise_min:self.action_bet_raise_all_in + 1] = True
                    
                    continue
            
            # Check if everyone has checked so far
            all_checked = (action_row_relevant_active == self.action_check).all().item()
            
            if all_checked:
                # Can check or bet (min bet + all bet sizes 6-16)
                legal_actions[elem][self.action_check] = True
                legal_actions[elem][self.action_min_bet:self.action_bet_raise_all_in + 1] = True
            else:
                # Someone has bet/raised - determine what we can do
                
                # Check if anyone has gone all-in (action 16)
                has_all_in = (action_row_relevant_active == self.action_bet_raise_all_in).any().item()
                
                # Check if someone has bet/raised (min_bet or any bet/raise action 5-16)
                has_bet_or_raise = (action_row_relevant_active >= self.action_min_bet).any().item()
                
                if has_bet_or_raise:
                    # Facing a bet - can fold or call
                    legal_actions[elem][self.action_fold] = True
                    legal_actions[elem][self.action_call] = True
                    
                    # Can we raise?
                    # Count number of raises this street (actions 6-16 when used as raises)
                    # Note: In no-limit poker, there's no raise cap - only limited by stacks
                    # We'll allow raises as long as player has chips
                    
                    if not has_all_in:
                        # Can raise if no all-in
                        # Need to check stack sizes
                        if player_stacks is not None:
                            player_stack = player_stacks[elem][next_position_to_act].item()
                            # Simple check: if stack > 0, can raise
                            if player_stack > self.big_blind:
                                legal_actions[elem][self.action_bet_raise_min:self.action_bet_raise_all_in + 1] = True
                        else:
                            # No stack info, assume can raise
                            legal_actions[elem][self.action_bet_raise_min:self.action_bet_raise_all_in + 1] = True
                    elif has_all_in:
                        # Facing all-in, can only fold or call (no re-raise)
                        pass  # Already set fold and call above
                else:
                    # No bet yet, but not all checks (edge case - shouldn't normally happen)
                    # Default to check and bet (all sizes)
                    legal_actions[elem][self.action_check] = True
                    legal_actions[elem][self.action_min_bet:self.action_bet_raise_all_in + 1] = True
                    
        return legal_actions
    
    def _is_hand_complete(
        self,
        active_players: torch.Tensor,
        all_in_players: typing.Optional[torch.Tensor]
    ) -> bool:
        """
        Check if the hand is complete (no more streets to play).
        
        Hand is complete when:
        1. Only 0 or 1 player is active (everyone else folded)
        2. All remaining players are all-in
        
        Args:
            active_players: Boolean mask of active players
            all_in_players: Boolean mask of all-in players (optional)
            
        Returns:
            True if hand is complete, False otherwise
        """
        num_active = active_players.sum().item()
        
        # If 0 or 1 player active, hand is over
        if num_active <= 1:
            return True
        
        # If all active players are all-in, hand is over (go to showdown)
        if all_in_players is not None:
            active_and_not_allin = active_players & (~all_in_players)
            if active_and_not_allin.sum().item() == 0:
                return True
        
        return False
    
    def _is_street_complete(
        self,
        action_row: torch.Tensor,
        position_row: torch.Tensor,
        active_players: torch.Tensor,
        all_in_players: typing.Optional[torch.Tensor],
        next_position: int,
        current_street: int
    ) -> bool:
        """
        Determine if betting is complete on this street.
        
        Street is complete when:
        1. All active non-all-in players have acted
        2. All bets/raises have been called or everyone folded
        3. Action has returned to a player who already acted with no uncalled bet
        
        Args:
            action_row: Actions taken this street
            position_row: Positions that acted this street
            active_players: Boolean mask of active (non-folded) players
            all_in_players: Boolean mask of all-in players (optional)
            next_position: Next position to act
            current_street: Current street index (0-3)
            
        Returns:
            True if street is complete, False otherwise
        """
        # Filter out inactive actions (folded, all-in, blinds, transitions)
        action_mask = (
            (action_row != self.action_inactive_folded) & 
            (action_row != self.action_inactive_allin) &
            (action_row != self.action_post_sb) &
            (action_row != self.action_post_bb) &
            (action_row != self.action_street_complete) &
            (action_row != self.action_hand_complete)
        )
        
        if action_mask.sum() == 0:
            # No real actions yet, street not complete
            return False
        
        actions_taken = action_row[action_mask]
        positions_acted = position_row[action_mask]
        
        # Identify players who can still act (active and not all-in)
        can_act = active_players.clone()
        if all_in_players is not None:
            can_act = can_act & (~all_in_players)
        
        num_can_act = can_act.sum().item()
        
        # Special case: only 0 or 1 player can act (everyone else folded/all-in)
        if num_can_act <= 1:
            return True
        
        # Check if all players who can act have acted this street
        positions_who_can_act = torch.where(can_act)[0]
        positions_who_acted = positions_acted.unique()
        
        all_acted = all(pos.item() in positions_who_acted for pos in positions_who_can_act)
        
        if not all_acted:
            # Not everyone has acted yet
            return False
        
        # Everyone has acted at least once. Now check if there's an uncalled bet/raise
        last_action = actions_taken[-1].item()
        
        # If last action was an aggressive action (bet/raise 5-16), check if it's been called
        if last_action >= self.action_min_bet and last_action <= self.action_bet_raise_all_in:
            # There's an uncalled bet/raise - street is NOT complete
            return False
        
        # If we reach here, either:
        # 1. Everyone checked (all actions are check)
        # 2. There was betting but it's been called (last action is call, fold, or check)
        
        # Additional check: make sure the next player to act has already acted
        # This prevents false positives when action comes back around
        if next_position in positions_acted:
            # Next player already acted this street
            # Check if there's been any betting since they last acted
            
            # Find last time next_position acted
            next_pos_actions = torch.where(positions_acted == next_position)[0]
            if len(next_pos_actions) > 0:
                last_action_idx = next_pos_actions[-1].item()
                
                # Check if there's been any aggressive action after this player last acted
                actions_after = actions_taken[last_action_idx + 1:]
                if len(actions_after) > 0:
                    any_aggressive_after = (actions_after >= self.action_min_bet).any().item()
                    if any_aggressive_after:
                        # There's been betting since this player acted - NOT complete
                        return False
        
        # Street is complete if we've made it here
        return True
    
    def _get_first_to_act_postflop(
        self,
        active_players: torch.Tensor,
        all_in_players: typing.Optional[torch.Tensor]
    ) -> typing.Optional[int]:
        """
        Determine first player to act post-flop.
        
        Post-flop, action starts with the first active non-all-in player
        after the button. In standard poker:
        - Position 0 = Button (in heads-up, this is also SB)
        - Position 1 = SB (in multi-way)
        - Position 2 = BB (in multi-way)
        
        For heads-up: Button/SB acts first post-flop (position 0)
        For multi-way: SB acts first post-flop (position 0 in our indexing where 0=SB)
        
        Args:
            active_players: Boolean mask of active players
            all_in_players: Boolean mask of all-in players (optional)
            
        Returns:
            Position index of first to act, or None if no one can act
        """
        # Determine who can act (active and not all-in)
        can_act = active_players.clone()
        if all_in_players is not None:
            can_act = can_act & (~all_in_players)
        
        if can_act.sum().item() == 0:
            return None
        
        # Post-flop, start from position 0 (SB in multi-way, BTN/SB in heads-up)
        # and find first player who can act
        for pos in range(self.num_players):
            if can_act[pos]:
                return pos
        
        return None

def get_min_bet_size_or_raise_size(
    pot_size_sequence,
    street_idxs,
    action_idxs,
    current_street,
):
    """
    Gets the minimum bet or raise size allowed in a position.
    Minimum allowable bet size is going to be 1 big blind or quarter pot.
    Minimum allowable raise size will be 1 big blinds or 1x the most recent increment, resulting in a bet size
    of either 2bb or 2 times the last increment.
    """
    
    current_street_idxs = street_idxs[street_idxs == current_street]

    if len(current_street_idxs) == 0:
        return max(2, max(pot_size_sequence.unique())/4)

    current_street_actions = action_idxs[street_idxs == current_street]

    if ((current_street_actions == 3).int() + (current_street_actions == 2).int()).sum() != current_street_actions.shape[0]:
        where_max = torch.where(pot_size_sequence == max(pot_size_sequence[0]))
        raise_size = pot_size_sequence[where_max[0][0], where_max[1][0]] - pot_size_sequence[where_max[0][0], where_max[1][0] - 1]
        return max([4, 2*raise_size])

class Player():
    """
    Encodes the player's current state.
    """
    def __init__(
        self,
        card_idxs : torch.Tensor,
        stack_size : float = 400.,
        position : int = 0,
        active_or_not : int = 1,
        device : str = "cpu"
    ):
        self.device = device
        self.stack_size = stack_size
        self.card_idxs = card_idxs
        self.position = position
        self.active_or_not = active_or_not

    def get_public_state(self):
        return self.stack_size, self.position, self.active_or_not

    def get_private_state(self):
        return self.get_public_state(), self.hole_cards

class Board():
    """
    Encodes the board state.
    """
    def __init__(
        self,
        board_cards : torch.Tensor,
        card_idxs : torch.Tensor
    ):
        self.board_cards = board_cards
        self.card_idxs = card_idxs