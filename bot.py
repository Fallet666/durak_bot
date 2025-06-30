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
    """Holds all information needed for decision making."""

    def __init__(self):
        self.trump: Optional[str] = None
        self.trump_card: Optional[Card] = None
        self.player: List[Card] = []
        self.opp_size: int = 6
        self.table = Table()
        self.discard: List[Card] = []
        self.turn = 'you'  # or 'opp'
        self.deck: int = 36

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
        """Probability that opponent has at least one card matching predicate."""
        pool = self.unseen()
        good = [c for c in pool if pred(c)]
        n = len(pool)
        k = len(good)
        m = min(self.opp_size, n)
        if n == 0 or m == 0 or k == 0:
            return 0.0
        from math import comb
        try:
            none_prob = comb(n - k, m) / comb(n, m)
        except ValueError:
            return 0.0
        return 1 - none_prob

    def sim_prob_opp(self, pred: Callable[[Card], bool], trials: int = 500) -> float:
        """Monte Carlo estimate of prob opponent has a card satisfying predicate."""
        pool = self.unseen()
        if not pool or self.opp_size == 0:
            return 0.0
        hits = 0
        sample_size = min(self.opp_size, len(pool))
        for _ in range(trials):
            hand = random.sample(pool, sample_size)
            if any(pred(c) for c in hand):
                hits += 1
        return hits / trials

    # ---------- cost ----------
    def cost(self, card: Card) -> int:
        base = Card.order.index(card.rank)
        return base + (9 if card.suit == self.trump else 0)

    def replenish(self):
        """Ask user about new cards from the deck and update state."""
        if self.deck <= 0:
            return
        try:
            mine = parse_cards(input('Ваш добор из колоды: ').split())
        except EOFError:
            mine = []
        self.player.extend(Card(c) for c in mine)
        try:
            opp_take = int(input('Сколько карт добрал соперник? ').strip() or '0')
        except (EOFError, ValueError):
            opp_take = 0
        self.opp_size += opp_take
        self.deck -= len(mine) + opp_take
        if self.deck < 0:
            self.deck = 0

    # ---------- decision ----------
    def best_attack(self) -> Optional[Card]:
        """Choose the attack card with minimal expected cost."""
        best, best_score = None, float('inf')
        for card in self.player:
            analytic = self.prob_opp(lambda x: x.beats(card, self.trump))
            sim = self.sim_prob_opp(lambda x: x.beats(card, self.trump))
            p_cover = (analytic + sim) / 2
            dup_bonus = -3 if sum(1 for c in self.player if c.rank == card.rank) > 1 else 0
            score = self.cost(card) * (1 + 2 * p_cover) + dup_bonus
            if score < best_score:
                best_score, best = score, card
        return best

    def defense_action(self) -> Tuple[str, Optional[str]]:
        ranks = {atk.rank for atk in self.table.unbeaten()}
        translate_opts = [c for c in self.player if c.rank in ranks]
        if translate_opts:
            best = min(translate_opts, key=self.cost)
            p_fail = (self.prob_opp(lambda x: x.beats(best, self.trump)) +
                      self.sim_prob_opp(lambda x: x.beats(best, self.trump))) / 2
            if p_fail < 0.5:
                return 'перевод', best.key

        if not self.table.unbeaten():
            return '', None

        best_pair = None
        best_score = float('inf')
        table_ranks = {atk.rank for atk, d in self.table.pairs}
        table_ranks.update(d.rank for _, d in self.table.pairs if d)
        for target in self.table.unbeaten():
            beaters = [c for c in self.player if c.beats(target, self.trump)]
            if not beaters:
                continue
            cand = min(beaters, key=self.cost)
            future_ranks = table_ranks | {target.rank, cand.rank}
            extra = (self.prob_opp(lambda x: x.rank in future_ranks) +
                     self.sim_prob_opp(lambda x: x.rank in future_ranks)) / 2
            score = self.cost(cand) * (1 + extra)
            if score < best_score:
                best_score = score
                best_pair = (target, cand)
        if best_pair:
            atk, dfn = best_pair
            return 'защита', f"{atk.key}>{dfn.key}"
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
    g.deck = 36 - len(g.player) - g.opp_size - 1

    while True:
        print('\n==== Состояние ====')
        print('Ваши:', ' '.join(str(c) for c in g.player))
        print('Стол :', g.table)
        print(f'Колода: {g.deck} | карт у оппа: {g.opp_size}')

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
        try:
            tokens = parse_cards(input('Карты оппа / пары: ').split())
        except EOFError:
            print('Нет ввода от оппонента. Завершение.')
            break
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
            g.replenish()
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
                g.replenish()
            else:
                g.turn = 'opp'


if __name__ == '__main__':
    main()
