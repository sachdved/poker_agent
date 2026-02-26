import torch
import typing
import numpy
from agent import *
from llm_modules import *
from ml_modules import *
from ml_ops_utils import *
from sequence_modules import *
from utils import *
import gc

def simulate_hand(
    num_players : int,
    street_idxs : torch.Tensor,
    table_position_idxs : torch.Tensor,
    action_idxs : torch.Tensor,
    pot_size_sequence : torch.Tensor,
    active_players : torch.Tensor,
    stack_size : torch.Tensor,
    table : typing.Dict,
    action_validator : PokerActionValidator,
    deck_order_shuffled : torch.Tensor,
):

    curr_street_index = 2
    curr_batch_index = 1
    END_OF_HAND_TOKEN_ACTIONS = 20
    END_OF_HAND_TOKEN_STREETS = 5
    
    END_OF_STREET_TOKEN_ACTIONS = 19
    END_OF_STREET_TOKEN_STREETS = 4

    softmax_prob = torch.nn.Softmax(dim=-1)
    
    end_of_hand_happened = False

    while not end_of_hand_happened:

        #print(f"after action {curr_street_index} \n\n")
    
        #print(curr_street_index)
        #print(curr_batch_index)

        #print(action_idxs[curr_batch_index, :curr_street_index])
        #print(pot_size_sequence[curr_batch_index, :curr_street_index])
        #print(table_position_idxs[curr_batch_index, :curr_street_index])
        #print(street_idxs[curr_batch_index, :curr_street_index])
        
        #print(stack_size[curr_batch_index])
        #print(active_players[curr_batch_index])
        
        #print("\n \n")
        
        legal_actions = action_validator.get_legal_actions_mask(
            street_idxs,
            table_position_idxs,
            action_idxs, 
            pot_size_sequence,
            active_players
        )
        
        who_is_acting = action_validator.get_next_to_act(
            street_idxs,
            table_position_idxs,
            action_idxs, 
            active_players
        )
        
        next_to_act = who_is_acting[[curr_batch_index-1]]
        #print(str(next_to_act.item()) + ' is the next to act')
        legal_actions = legal_actions[curr_batch_index-1]
        
        if legal_actions[ -1] == True:
            # End of hand has happened. Terminate the sequence and end while loop.
            end_of_hand_happened = True
            
            action_idxs[curr_batch_index:, curr_street_index] = END_OF_HAND_TOKEN_ACTIONS
            street_idxs[curr_batch_index:, curr_street_index] = END_OF_HAND_TOKEN_STREETS
            pot_size_sequence[curr_batch_index: , curr_street_index] = pot_size_sequence[curr_batch_index, curr_street_index - 1]
            table_position_idxs[curr_batch_index: , curr_street_index] = num_players
        
        elif legal_actions[-2] == True:
            # End of street has happened. Prepare transition to next street.    
            action_idxs[curr_batch_index:, curr_street_index] = END_OF_STREET_TOKEN_ACTIONS
            street_idxs[curr_batch_index:, curr_street_index] = END_OF_STREET_TOKEN_STREETS
            pot_size_sequence[curr_batch_index: , curr_street_index] = pot_size_sequence[curr_batch_index, curr_street_index - 1]
            table_position_idxs[curr_batch_index: , curr_street_index] = num_players
        
            if street_idxs[curr_batch_index, curr_street_index - 1] == 1:
                current_street = "flop"
            elif street_idxs[curr_batch_index, curr_street_index - 1] == 2:
                current_street = "turn"
            elif street_idxs[curr_batch_index, curr_street_index - 1] == 3:
                current_street = "river"
            
        
            for key in table.keys():
                _, cards, _ = table[key]
                if current_street == "flop":
                    # Deal three cards (OMG NO BURN?!).
                    cards[curr_batch_index:, 0, 2:5] = deck_order_shuffled[0,(2*(num_players)):(2*(num_players)+3)]%13
                    cards[curr_batch_index:, 1, 2:5] = deck_order_shuffled[0,(2*(num_players)):(2*(num_players)+3)]//13
                    table[key][1] = cards
                elif current_street == "turn":
                    # Deal turn card (OMG NO BURN?!).
                    cards[curr_batch_index:, 0, 5] = deck_order_shuffled[0,(2*(num_players)+3)]%13
                    cards[curr_batch_index:, 1, 5] = deck_order_shuffled[0,(2*(num_players)+3)]//13
                    table[key][1] = cards
                elif current_street == "river":
                    # Deal river card (still no burn).
                    cards[curr_batch_index:, 0, 6] = deck_order_shuffled[0,(2*(num_players)+4)]%13
                    cards[curr_batch_index:, 1, 6] = deck_order_shuffled[0,(2*(num_players)+4)]//13
                    table[key][1] = cards
    
            if (action_idxs[curr_batch_index]==16).any():
                # Someone was all in on the street that just completed, implying the hand is over.
                action_idxs[curr_batch_index:, curr_street_index + 1] = END_OF_HAND_TOKEN_ACTIONS
                street_idxs[curr_batch_index:, curr_street_index + 1] = END_OF_HAND_TOKEN_STREETS
                pot_size_sequence[curr_batch_index: , curr_street_index + 1] = pot_size_sequence[curr_batch_index, curr_street_index - 1]
                table_position_idxs[curr_batch_index: , curr_street_index + 1] = num_players
    
                for key in table.keys():
                    _, cards, _ = table[key]
                    cards[curr_batch_index:, 0, 2:7] = deck_order_shuffled[0,(2*(num_players)):(2*(num_players)+5)]%13
                    cards[curr_batch_index:, 1, 2:7] = deck_order_shuffled[0,(2*(num_players)):(2*(num_players)+5)]//13
                    table[key][1] = cards
                
                end_of_hand_happened = True
        
        else:
            player, cards, position = table[next_to_act.item()]
    
            cards = cards[[curr_batch_index]]
            position = position[[curr_batch_index]]
        
            outputs = player(
                position,
                cards,
                street_idxs[[curr_batch_index]],
                table_position_idxs[[curr_batch_index]],
                action_idxs[[curr_batch_index]],
                pot_size_sequence[[curr_batch_index]],
                active_players.to('cuda')[[curr_batch_index]],
                stack_size.to('cuda')[[curr_batch_index]]
            )

            #print(outputs['probits'])
            masked_logits = outputs['probits'].masked_fill(~legal_actions.to('cuda'), float('-inf'))
            sampled_action = torch.distributions.Categorical(softmax_prob(masked_logits)).sample()
    
            if street_idxs[curr_batch_index, curr_street_index - 1]!=4:
                current_street = street_idxs[curr_batch_index, curr_street_index - 1] 
            else:
                current_street = street_idxs[curr_batch_index, curr_street_index - 2] + 1
    
    
    
            if 5 <= sampled_action <= 16:
                min_size = get_min_bet_size_or_raise_size(
                    pot_size_sequence[[curr_batch_index]],
                    street_idxs[[curr_batch_index]],
                    action_idxs[[curr_batch_index]],
                    current_street
                )
            
                if legal_actions[5] == True:
                    spacing = 16-5+1 # (number of sizes)
                else:
                    spacing = 16-6+1 # (number of sizes)
                
                my_stack_size = stack_size[curr_batch_index, position.int().to('cpu')].squeeze()

                #print(min_size)
                #print(my_stack_size)
                
                multiplicative_factor = (my_stack_size / min(min_size, my_stack_size)) ** (1/(spacing-1))
                
                bet_sizes = torch.ones(spacing) * min(min_size, my_stack_size)
                bet_sizes = torch.ceil(bet_sizes * (multiplicative_factor ** torch.arange(spacing))).clip(max = my_stack_size)
                
                chosen_bet = bet_sizes[sampled_action.to('cpu')[0] - 17]
            
                street_idxs[curr_batch_index:, curr_street_index] = current_street
                table_position_idxs[curr_batch_index:, curr_street_index] = position.int().to('cpu')
    
                action_idxs[curr_batch_index:, curr_street_index] = sampled_action.to('cpu')[0]
                pot_size_sequence[curr_batch_index:, curr_street_index] = (
                    pot_size_sequence[curr_batch_index, curr_street_index - 1]
                    +
                    bet_sizes[sampled_action.to('cpu')[0] - 17]
                )
                stack_size[curr_batch_index:, position.int().to('cpu')] -= bet_sizes[sampled_action.to('cpu')[0] - 17]
    
                if stack_size[curr_batch_index, position.int().to('cpu')] == 0:
                    # if we're left with nothing left, it's an all in.
                    action_idxs[curr_batch_index:, curr_street_index] = 16
                
    
            if sampled_action == 4:
                
                two_largest_vals, _ = torch.topk(
                    torch.unique(pot_size_sequence[curr_batch_index]), 2
                )
    
                # The amount to call, if everyone started with the same amount
                # of money is equal to my stack size minus the smallest stack size
                # because that person made the biggest bet so far.
                amount_to_call = min(
                    stack_size[curr_batch_index, position.int().to('cpu')] - min(stack_size[curr_batch_index]), 
                    stack_size[curr_batch_index, position.int().to('cpu')]
                )
    
                street_idxs[curr_batch_index:, curr_street_index] = current_street
                table_position_idxs[curr_batch_index:, curr_street_index] = position.int().to('cpu')
                action_idxs[curr_batch_index:, curr_street_index] = 4
                pot_size_sequence[curr_batch_index:, curr_street_index] = (
                    pot_size_sequence[curr_batch_index, curr_street_index - 1] + amount_to_call
                )
                stack_size[curr_batch_index:, position.int().to('cpu')] -= amount_to_call
    
            if sampled_action == 3:
    
                street_idxs[curr_batch_index:, curr_street_index] = current_street
                table_position_idxs[curr_batch_index:, curr_street_index] = position.int().to('cpu')
                action_idxs[curr_batch_index:, curr_street_index] = 3
                pot_size_sequence[curr_batch_index:, curr_street_index] = (
                    max(pot_size_sequence[curr_batch_index])
                )
    
            if sampled_action == 2:
    
                street_idxs[curr_batch_index:, curr_street_index] = current_street
                table_position_idxs[curr_batch_index:, curr_street_index] = position.int().to('cpu')
                action_idxs[curr_batch_index:, curr_street_index] = 2
                pot_size_sequence[curr_batch_index:, curr_street_index] = (
                    pot_size_sequence[curr_batch_index, curr_street_index - 1]
                )
    
                active_players[curr_batch_index:, position.int().to('cpu')] = 0
    
            del outputs
            gc.collect()
            torch.cuda.empty_cache()
    
            
            
        curr_street_index += 1
        curr_batch_index += 1

    return (
        street_idxs[:curr_batch_index], 
        table_position_idxs[:curr_batch_index], 
        action_idxs[:curr_batch_index], 
        pot_size_sequence[:curr_batch_index], 
        active_players[:curr_batch_index], 
        stack_size[:curr_batch_index],
        table
    )