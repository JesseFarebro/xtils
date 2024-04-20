import math
from typing import Dict, NamedTuple

from xtils.domains.atari.environment import AtariGame


class AtariBaselineScores(NamedTuple):
    human: float
    random: float


_SCORES: Dict[str, AtariBaselineScores] = {
    "Alien": AtariBaselineScores(human=7127.7, random=227.8),
    "Amidar": AtariBaselineScores(human=1719.5, random=5.8),
    "Assault": AtariBaselineScores(human=742.0, random=222.4),
    "Asterix": AtariBaselineScores(human=8503.3, random=210.0),
    "Asteroids": AtariBaselineScores(human=47388.7, random=719.1),
    "Atlantis": AtariBaselineScores(human=29028.1, random=12850.0),
    "BankHeist": AtariBaselineScores(human=753.1, random=14.2),
    "BattleZone": AtariBaselineScores(human=37187.5, random=2360.0),
    "BeamRider": AtariBaselineScores(human=16926.5, random=363.9),
    "Berzerk": AtariBaselineScores(human=2630.4, random=123.7),
    "Bowling": AtariBaselineScores(human=160.7, random=23.1),
    "Boxing": AtariBaselineScores(human=12.1, random=0.1),
    "Breakout": AtariBaselineScores(human=30.5, random=1.7),
    "Centipede": AtariBaselineScores(human=12017.0, random=2090.9),
    "ChopperCommand": AtariBaselineScores(human=7387.8, random=811.0),
    "CrazyClimber": AtariBaselineScores(human=35829.4, random=10780.5),
    "Defender": AtariBaselineScores(human=18688.9, random=2874.5),
    "DemonAttack": AtariBaselineScores(human=1971.0, random=152.1),
    "DoubleDunk": AtariBaselineScores(human=-16.4, random=-18.6),
    "Enduro": AtariBaselineScores(human=860.5, random=0.0),
    "FishingDerby": AtariBaselineScores(human=-38.7, random=-91.7),
    "Freeway": AtariBaselineScores(human=29.6, random=0.0),
    "Frostbite": AtariBaselineScores(human=4334.7, random=65.2),
    "Gopher": AtariBaselineScores(human=2412.5, random=257.6),
    "Gravitar": AtariBaselineScores(human=3351.4, random=173.0),
    "Hero": AtariBaselineScores(human=30826.4, random=1027.0),
    "IceHockey": AtariBaselineScores(human=0.9, random=-11.2),
    "Jamesbond": AtariBaselineScores(human=302.8, random=29.0),
    "Kangaroo": AtariBaselineScores(human=3035.0, random=52.0),
    "Krull": AtariBaselineScores(human=2665.5, random=1598.0),
    "KungFuMaster": AtariBaselineScores(human=22736.3, random=258.5),
    "MontezumaRevenge": AtariBaselineScores(human=4753.3, random=0.0),
    "MsPacman": AtariBaselineScores(human=6951.6, random=307.3),
    "NameThisGame": AtariBaselineScores(human=8049.0, random=2292.3),
    "Phoenix": AtariBaselineScores(human=7242.6, random=761.4),
    "Pitfall": AtariBaselineScores(human=6463.7, random=-229.4),
    "Pong": AtariBaselineScores(human=14.6, random=-20.7),
    "PrivateEye": AtariBaselineScores(human=69571.3, random=24.9),
    "Qbert": AtariBaselineScores(human=13455.0, random=163.9),
    "Riverraid": AtariBaselineScores(human=17118.0, random=1338.5),
    "RoadRunner": AtariBaselineScores(human=7845.0, random=11.5),
    "Robotank": AtariBaselineScores(human=11.9, random=2.2),
    "Seaquest": AtariBaselineScores(human=42054.7, random=68.4),
    "Skiing": AtariBaselineScores(human=-4336.9, random=-17098.1),
    "Solaris": AtariBaselineScores(human=12326.7, random=1236.3),
    "SpaceInvaders": AtariBaselineScores(human=1668.7, random=148.0),
    "StarGunner": AtariBaselineScores(human=10250.0, random=664.0),
    "Surround": AtariBaselineScores(human=6.5, random=-10.0),
    "Tennis": AtariBaselineScores(human=-8.3, random=-23.8),
    "TimePilot": AtariBaselineScores(human=5229.2, random=3568.0),
    "Tutankham": AtariBaselineScores(human=167.6, random=11.4),
    "UpNDown": AtariBaselineScores(human=11693.2, random=533.4),
    "Venture": AtariBaselineScores(human=1187.5, random=0.0),
    "VideoPinball": AtariBaselineScores(human=17667.9, random=16256.9),
    "WizardOfWor": AtariBaselineScores(human=4756.5, random=563.5),
    "YarsRevenge": AtariBaselineScores(human=54576.9, random=3092.9),
    "Zaxxon": AtariBaselineScores(human=9173.3, random=32.5),
}


def atari_baseline_scores(game: AtariGame) -> AtariBaselineScores:
    """Returns the human and random scores for the given Atari game."""
    return _SCORES[game]


def human_normalized_score(game: AtariGame, raw_score: float) -> float:
    """Converts game score to human-normalized score."""
    if game not in _SCORES:
        raise ValueError(f"Game {game} doesn't have baseline scores.")
    scores = _SCORES[game]
    return (raw_score - scores.random) / (scores.human - scores.random)


__all__ = ["atari_baseline_scores", "human_normalized_score"]
