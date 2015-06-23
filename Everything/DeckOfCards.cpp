//
//  DeckOfCards.cpp
//  Everything
//
//  Created by Xing Zhao on 6/18/15.
//  Copyright (c) 2015 Zhao, Xing. All rights reserved.
//

#include "DeckOfCards.h"
enum Suit { SPADES, CLUBS, HEARTS, DIAMONDS, };
enum Face { ACE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING, };

class Card {
private:
    Suit suit;
    Face face;
public:
    Card() {};
    Card(Suit suit, Face face) : suit(suit), face(face) {}
    Card(const Card& orig) : suit(orig.suit), face(orig.face) {}
    Suit getSuit() const { return suit; }
    Face getFace() const { return face; }
};

class Deck {
private:
    Card cards[52];
public:
    Deck() {
        int index = 0;
        for (int i = 0; i < SUITS_PER_DECK; ++i) {
            for (int j = 0; j < CARDS_PER_SUIT; ++j) {
                index = i * CARDS_PER_SUIT + j;
                cards[index] = Card((Suit) i, (Face)j);
            }
        }
    }
    
    Deck(const Deck& orig) {
        for (int i = 0; i < SUITS_PER_DECK * CARDS_PER_SUIT; ++i) {
            cards[i] = orig.cards[i];
        }
    }
    
    void Shuffle() {
        int bagSize = SUITS_PER_DECK * CARDS_PER_SUIT;
        int index = 0;
        srand(time(NULL));
        while (bagSize) {
            index = rand() % bagSize;
            swap(cards[--bagSize], cards[index]);
        }
    }
    
    static const int SUITS_PER_DECK = 4;
    static const int CARDS_PER_SUIT = 13;

};