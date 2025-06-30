import random
from typing import List, Optional, Tuple, Callable

SUITS = ['H', 'D', 'C', 'S']
RANKS = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
ALL_CARDS = [r + s for s in SUITS for r in RANKS]

# Пользовательские обозначения мастей ➜ внутренние
SUIT_MAP = {'L': 'H', 'R': 'D', 'K': 'C', 'S': 'S'}
DISPLAY = {'H': '♥', 'D': '♦', 'C': '♣', 'S': '♠'}


class Card:
    order = RANKS

    def __init__(self, val: str):
        val = val.upper()
        self.rank, raw_suit = val[:-1], val[-1]
        self.suit = SUIT_MAP.get(raw_suit, raw_suit)
        self.key = self.rank + self.suit

    def __str__(self):
        return f"{self.rank}{DISPLAY[self.suit]}"

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.key == other.key

    def beats(self, other: 'Card', trump: str) -> bool:
        if self.suit == other.suit:
            return Card.order.index(self.rank) > Card.order.index(other.rank)
        return self.suit == trump and other.suit != trump


class Table:
    """Keeps attack/defense pairs."""

    def __init__(self):
        self.pairs: List[Tuple[Card, Optional[Card]]] = []  # (attack, defense)

    def add_attack(self, c: Card):
        self.pairs.append((c, None))

    def add_defense(self, attack: Card, defense: Card):
        # replace first unmatched attack with same card
        for i, (atk, dfn) in enumerate(self.pairs):
            if dfn is None and atk.key == attack.key:
                self.pairs[i] = (atk, defense)
                return True
        return False  # not found

    def unbeaten(self) -> List[Card]:
        return [atk for atk, dfn in self.pairs if dfn is None]

    def clear(self) -> List[Card]:
        cleared = []
        for atk, dfn in self.pairs:
            cleared.append(atk)
            if dfn:
                cleared.append(dfn)
        self.pairs.clear()
        return cleared

    def __str__(self):
        seg = []
        for atk, dfn in self.pairs:
            seg.append(f"{atk}{'>' + str(dfn) if dfn else ''}")
        return ' '.join(seg)


class GameState:
    def __init__(self):
        self.trump: Optional[str] = None
        self.trump_card: Optional[Card] = None
        self.player: List[Card] = []
        self.opp_size: int = 6
        self.table = Table()
        self.discard: List[Card] = []
        self.turn = 'you'  # or 'opp'

    # ---------- util ----------
    def unseen(self) -> List[Card]:
        seen = {c.key for c in self.player}
        seen.update(c.key for c in self.discard)
        for atk, dfn in self.table.pairs:
            seen.add(atk.key)
            if dfn:
                seen.add(dfn.key)
        seen.add(self.trump_card.key)
        return [Card(k) for k in ALL_CARDS if k not in seen]

    def prob_opp(self, pred: Callable[[Card], bool]) -> float:
        pool = self.unseen()
        good = [c for c in pool if pred(c)]
        if not pool or self.opp_size == 0:
            return 0.0
        return min(1.0, len(good) / len(pool) * self.opp_size / len(pool))

    # ---------- cost ----------
    def cost(self, card: Card) -> int:
        base = Card.order.index(card.rank)
        return base + (9 if card.suit == self.trump else 0)

    # ---------- decision ----------
    def best_attack(self) -> Optional[Card]:
        best, best_score = None, 1e9
        for card in self.player:
            p_cover = self.prob_opp(lambda x: x.beats(card, self.trump))
            score = p_cover * 25 + self.cost(card)
            if score < best_score:
                best_score, best = score, card
        return best

    def defense_action(self) -> Tuple[str, Optional[str]]:
        # try translate
        ranks = {atk.rank for atk in self.table.unbeaten()}
        translate_opts = [c for c in self.player if c.rank in ranks]
        if translate_opts:
            best = min(translate_opts, key=self.cost)
            return 'перевод', best.key
        # else defend cheapest first unbeaten
        if not self.table.unbeaten():
            return '', None
        target = self.table.unbeaten()[0]
        beaters = [c for c in self.player if c.beats(target, self.trump)]
        if beaters:
            best = min(beaters, key=self.cost)
            return 'защита', f"{target.key}>{best.key}"
        return 'взятие', None


# ----------- helpers -------------

def parse_cards(tokens: List[str]) -> List[str]:
    res = []
    for t in tokens:
        if '>' in t:  # defense pair keep whole
            res.append(t)
        else:
            res.extend(t.split(','))
    return [s for s in res if s]


# ----------- CLI -------------

def main():
    g = GameState()
    g.trump_card = Card(input('Козырная карта: ').strip())
    g.trump = g.trump_card.suit
    g.player = [Card(c) for c in parse_cards(input('Ваши карты: ').split())]
    g.turn = 'you' if input('Кто первым? (1-вы 2-опп): ').strip() == '1' else 'opp'

    while True:
        print('\n==== Состояние ====')
        print('Ваши:', ' '.join(str(c) for c in g.player))
        print('Стол :', g.table)

        if g.turn == 'you':
            atk = g.best_attack()
            if not atk:
                print('Нечем ходить')
                break
            print('Бот ходит:', atk)
            g.table.add_attack(atk)
            g.player.remove(atk)
            g.turn = 'opp'
            continue

        # ход оппонента: пользователь вводит действия
        tokens = parse_cards(input('Карты оппа / пары: ').split())
        for tok in tokens:
            if '>' in tok:  # defense pair
                atk_s, def_s = tok.split('>')
                g.table.add_defense(Card(atk_s), Card(def_s))
            else:  # new attack
                g.table.add_attack(Card(tok))

        action, payload = g.defense_action()
        if action == 'взятие':
            print('Бот берёт все карты')
            g.player.extend(g.table.clear())
            g.turn = 'you'
            continue
        if action == 'перевод':
            c = Card(payload)
            print('Бот переводит:', c)
            g.table.add_attack(c)
            g.player.remove(c)
            g.turn = 'opp'
            continue
        if action == 'защита':
            atk_s, def_s = payload.split('>')
            print('Бот кроет:', f"{Card(atk_s)}>{Card(def_s)}")
            g.table.add_defense(Card(atk_s), Card(def_s))
            g.player.remove(Card(def_s))
            # check if all covered and opp passed
            if not g.table.unbeaten():
                g.discard.extend(g.table.clear())
                g.turn = 'you'
            else:
                g.turn = 'opp'


if __name__ == '__main__':
    main()
