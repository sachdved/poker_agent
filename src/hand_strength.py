import torch
from collections import defaultdict
import typing

def determine_winner(
    active_players : torch.Tensor,
    pot_size_sequence : torch.Tensor,
    stack_sizes : torch.Tensor,
    table : typing.Dict,
    starting_stack_size : int = 400,
):
    """
    Determines the winner of a hand. and assigns the reward. First, identifies if someone won because of folds.
    Next, gets hand strength of each player and ranks hands. Returns the rewards for each player.
    """

    # Base reward is how much you've put into the pot.

    reward = stack_sizes - starting_stack_size
    total_pot = max(pot_size_sequence)

    if active_players.sum() == 1:
        winner = torch.where(active_players)[0].item()
        reward[winner] += total_pot

    else:
        hand_strengths = []
        for key in table.keys():
            hand_strengths.append(
                hand_strength(table[key][1][-1]) # get the hand strength of the final state of the hand.
            )
        winning_hand = max(hand_strengths)
        winners = [index for index, hand in enumerate(hand_strengths) if hand == winning_hand]
        reward[winners] += total_pot/len(winners)
    return reward

def hand_strength(
    hand: torch.Tensor
):
    """
    Takes in a 7-card hand and returns the hand strength of the best possible five-card hand from
    the 7-card hand, and its strength. Hand strength is returned as a 6-tuple, where the first number is as follows:
    0: high card
    1: 1 pair
    2: 2 pair
    3: three of a kind
    4: straight
    5: flush
    6: full house
    7: four of a kind
    8: straight flush

    In order to deal with ties, the second number will represent the strength of the 'strongest' card in the hand. 
    To give an example, if we imagine we have an Ace-high flush, we will report our hand as (5, 13), indicating flush, 
    ace high. The second number will represent the strength of the second strongest card. In the case of the 1 paired hands, this
    is the same as the strength of the strongest card. We do this all the way down through the strength of the hand. 
    The reason to do this is to determine who wins in the case of tie breaks.

    In the case of straights, they can be ace low or ace high. Ace high straights would be denoted (4, 12, 11, 10, 9, 8)
    and ace low straights would be denoted (4, 3, 2, 1, 0, 12).

    12: Ace
    11: King
    10: Queen
    9: Jack
    8: 10
    7: 9
    6: 8
    5: 7
    4: 6
    3: 5
    2: 4
    1: 3
    0: 2.
    
    This function will largely be written as a series of if statements, I suppose. I'm not sure how else to think through the 
    complexity of the hand. 
    """
    rank_counter = defaultdict(int)
    suit_counter = defaultdict(int)

    ranks = hand[0, :]
    suits = hand[1, :]
    for i in range(7):
        rank_counter[ranks[i].item()] += 1
        suit_counter[suits[i].item()] += 1

    ## check for straight flush
    flush, flush_ranks = _is_flush(hand)
    flush_ranks = sorted(flush_ranks)
    if flush:
        straight, straight_ranks = _is_straight(flush_ranks)
        if straight:
            return (8, straight_ranks[0], straight_ranks[1], straight_ranks[2], straight_ranks[3], straight_ranks[4])
                
    ## check for 4 of a kind
    quad_rank = [r for r in rank_counter.keys() if rank_counter[r] == 4]
    if len(quad_rank) != 0:
        kicker = sorted([r for r in rank_counter.keys() if rank_counter[r] != 4])
        return (7, quad_rank[0], quad_rank[0], quad_rank[0], quad_rank[0], kicker[-1])

    ## check for full houses
    trips_rank = [r for r in rank_counter.keys() if rank_counter[r] == 3] 
    doubles_rank = [r for r in rank_counter.keys() if rank_counter[r] >= 2]

    if len(trips_rank) != 0 and len(doubles_rank) != 0:
        trip_rank = max(trips_rank)
        doubles_rank_trip_removed = [k for k in doubles_rank if k != trip_rank]
        if len(doubles_rank_trip_removed) > 0:
            double_rank = max(doubles_rank_trip_removed)
            return (6, trip_rank, trip_rank, trip_rank, double_rank, double_rank)
    
    ## check for flush but not straight
    if flush and not straight:
        return (5, flush_ranks[-1], flush_ranks[-2], flush_ranks[-3], flush_ranks[-4], flush_ranks[-5])

    ## check for straight but not flush
    ranks = list(set(list(rank_counter.keys())))
    sorted_ranks = sorted(ranks)
    straight, straight_ranks = _is_straight(sorted_ranks)
    if straight:
        return (4, straight_ranks[0], straight_ranks[1], straight_ranks[2], straight_ranks[3], straight_ranks[4])
            
    ## check for 3 of a kind
    if len(trips_rank)==1:
        kickers = sorted([k for k in rank_counter.keys() if rank_counter[k] == 1])
        trip_rank = trips_rank[0]
        return (3, trip_rank, trip_rank, trip_rank, kickers[-1], kickers[-2])

    ## check for 2 pair
    if len(doubles_rank) >= 2:
        doubles_rank = sorted(doubles_rank)
        pair_1 = doubles_rank[-1]
        pair_2 = doubles_rank[-2]
        kicker = max([k for k in rank_counter.keys() if k != pair_1 and k != pair_2])
        return (2, pair_1, pair_1, pair_2, pair_2, kicker)

    ## check for 1 pair
    if len(doubles_rank) == 1:
        pair = doubles_rank[0]
        kickers = sorted([k for k in rank_counter.keys() if k != pair])
        return (1, pair, pair, kickers[-1], kickers[-2], kickers[-3])

    ## Return high card hand
    ranks = sorted(list(rank_counter.keys()))
    return (0, ranks[-1], ranks[-2], ranks[-3], ranks[-4], ranks[-5])

def _is_flush(hand):
    """
    Checks if the hand is a flush. This is written
    to reduce some of the code duplication in the hand
    strength computation. 

    Returns whether it is a flush or not, and the cards
    belong to the suit creating the flush.
    """
    ranks = hand[0, :]
    suits = hand[1, :]
    suit_counter = defaultdict(int)
    for i in range(7):
        suit_counter[suits[i].item()] += 1
    for key in suit_counter.keys():
        if suit_counter[key] >= 5:
            ranks_flush = []
            for i in range(7):
                if suits[i].item() == key:
                    ranks_flush.append(ranks[i].item())
            return True, ranks
    return False, []

def _is_straight(sorted_ranks):
    """
    Checks if the hand is a straight, given the ranks in sorted order.

    Returns the highest possible straight, given the ranks, if possible.
    """
    sorted_ranks = list(reversed(sorted_ranks))
    for i in range(len(sorted_ranks)-4):
        if sorted_ranks[i] - sorted_ranks[i+4] == 4:
            return True, (sorted_ranks[i:i+5])

    if sorted_ranks[0] == 12:
        if sorted_ranks[-4:] == [3, 2, 1, 0]:
            return True, [3, 2, 1, 0, 12]
    return False, []