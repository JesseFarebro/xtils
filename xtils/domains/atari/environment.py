import dataclasses
import enum
import typing
from typing import Literal, Mapping, NotRequired, Sequence, TypedDict

import ale_py
import dm_env
import jax
import numpy as np
import numpy.typing as npt
from ale_py import roms
from dm_env import specs
from dm_env.auto_reset_environment import AutoResetEnvironment

from xtils.domains.atari import metadata as game_metadata

Atari57Game = Literal[
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "Asteroids",
    "Atlantis",
    "BankHeist",
    "BattleZone",
    "BeamRider",
    "Berzerk",
    "Bowling",
    "Boxing",
    "Breakout",
    "Centipede",
    "ChopperCommand",
    "CrazyClimber",
    "Defender",
    "DemonAttack",
    "DoubleDunk",
    "Enduro",
    "FishingDerby",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Gravitar",
    "Hero",
    "IceHockey",
    "Jamesbond",
    "Kangaroo",
    "Krull",
    "KungFuMaster",
    "MontezumaRevenge",
    "MsPacman",
    "NameThisGame",
    "Phoenix",
    "Pitfall",
    "Pong",
    "PrivateEye",
    "Qbert",
    "Riverraid",
    "RoadRunner",
    "Robotank",
    "Seaquest",
    "Skiing",
    "Solaris",
    "SpaceInvaders",
    "StarGunner",
    "Surround",
    "Tennis",
    "TimePilot",
    "Tutankham",
    "UpNDown",
    "Venture",
    "VideoPinball",
    "WizardOfWor",
    "YarsRevenge",
    "Zaxxon",
]
AtariGame = Literal[
    "Adventure",
    "AirRaid",
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "Asteroids",
    "Atlantis2",
    "Atlantis",
    "Backgammon",
    "BankHeist",
    "BasicMath",
    "BattleZone",
    "BeamRider",
    "Berzerk",
    "Blackjack",
    "Bowling",
    "Boxing",
    "Breakout",
    "Carnival",
    "Casino",
    "Centipede",
    "ChopperCommand",
    "CrazyClimber",
    "Crossbow",
    "Darkchambers",
    "Defender",
    "DemonAttack",
    "DonkeyKong",
    "DoubleDunk",
    "Earthworld",
    "ElevatorAction",
    "Enduro",
    "Entombed",
    "Et",
    "FishingDerby",
    "FlagCapture",
    "Freeway",
    "Frogger",
    "Frostbite",
    "Galaxian",
    "Gopher",
    "Gravitar",
    "Hangman",
    "HauntedHouse",
    "Hero",
    "HumanCannonball",
    "IceHockey",
    "Jamesbond",
    "JourneyEscape",
    "Kaboom",
    "Kangaroo",
    "KeystoneKapers",
    "KingKong",
    "Klax",
    "Koolaid",
    "Krull",
    "KungFuMaster",
    "LaserGates",
    "LostLuggage",
    "MarioBros",
    "MiniatureGolf",
    "MontezumaRevenge",
    "MrDo",
    "MsPacman",
    "NameThisGame",
    "Othello",
    "Pacman",
    "Phoenix",
    "Pitfall2",
    "Pitfall",
    "Pong",
    "Pooyan",
    "PrivateEye",
    "Qbert",
    "Riverraid",
    "RoadRunner",
    "Robotank",
    "Seaquest",
    "SirLancelot",
    "Skiing",
    "Solaris",
    "SpaceInvaders",
    "SpaceWar",
    "StarGunner",
    "Superman",
    "Surround",
    "Tennis",
    "Tetris",
    "TicTacToe3D",
    "TimePilot",
    "Trondead",
    "Turmoil",
    "Tutankham",
    "UpNDown",
    "Venture",
    "VideoCheckers",
    "VideoChess",
    "VideoCube",
    "VideoPinball",
    "WizardOfWor",
    "WordZapper",
    "YarsRevenge",
    "Zaxxon",
]


@enum.unique
class ActionSet(enum.Enum):
    Complete = enum.auto()
    Minimal = enum.auto()


@enum.unique
class ObservationType(enum.Enum):
    ImageRGB = "rgb"
    ImageGrayscale = "grayscale"
    RAM = "ram"
    Lives = "lives"
    Annotations = "annotations"


@enum.unique
class RenderMode(enum.Enum):
    Human = enum.auto()


@enum.unique
class FrameProcessing(enum.Enum):
    ColorAveraging = enum.auto()


@dataclasses.dataclass(frozen=True)
class AtariEnvironmentConfig:
    _: dataclasses.KW_ONLY
    game: AtariGame
    mode: int | None = None
    difficulty: int | None = None
    seed: int | Sequence[int] | np.random.SeedSequence | None = None
    repeat_action_probability: float = 0.25
    frameskip: int = 1
    max_episode_frames: int = 108_000
    render_mode: RenderMode | None = None
    frame_processing: FrameProcessing | None = None
    action_set: ActionSet = ActionSet.Minimal
    observation_type: ObservationType | Sequence[ObservationType] | None = dataclasses.field(
        default_factory=lambda: [
            ObservationType.ImageRGB,
            ObservationType.Lives,
        ]
    )

    def replace(self, **kwargs) -> "AtariEnvironmentConfig":
        return dataclasses.replace(self, **kwargs)

    def setup(self) -> None:
        if self.game not in typing.get_args(AtariGame):
            raise ValueError(f"Game {self.game} is not supported.")

        if not isinstance(self.frameskip, int):
            raise ValueError(f"Invalid frameskip type {type(self.frameskip)}. Expected `int`.")
        if self.frameskip < 1:
            raise ValueError("Frameskip must be an intger that is >= 1")

        if self.seed is not None and type(self.seed) not in (
            int,
            Sequence,
            np.random.SeedSequence,
        ):
            raise ValueError("seed must be of type `int` or `Sequence[int]` " "or `np.random.SeedSequence`")
        if self.repeat_action_probability < 0 or self.repeat_action_probability > 1:
            raise ValueError("repeat_action_probability must be in [0, 1]")

        if self.max_episode_frames <= 0:
            raise ValueError("Max episode frames must be positive.")

        if self.render_mode is not None and not isinstance(self.render_mode, RenderMode):
            raise ValueError("render_mode must be of type `RenderMode`")

    def interface(self) -> tuple[ale_py.ALEInterface, Sequence[ale_py.Action]]:
        ale = ale_py.ALEInterface()

        seed = self.seed
        if seed is None or not isinstance(seed, np.random.SeedSequence):
            seed = np.random.SeedSequence(self.seed)  # type: ignore
        ale.setInt("random_seed", seed.generate_state(1).astype(np.int32))  # type: ignore

        # Stochasticity settings
        ale.setFloat("repeat_action_probability", self.repeat_action_probability)

        # Frame skip
        ale.setInt("frame_skip", self.frameskip)

        # Display mode
        ale.setBool("display_screen", self.render_mode == RenderMode.Human)
        ale.setBool("sound", self.render_mode == RenderMode.Human)

        # Truncation settings
        ale.setInt("max_num_frames_per_episode", self.max_episode_frames)

        # Load ROM
        ale.loadROM(getattr(roms, self.game))

        if self.mode is not None:
            if self.mode not in ale.getAvailableModes():
                raise ValueError(f"Invalid mode {self.mode}. Available modes: {ale.getAvailableModes()}.")
            ale.setMode(self.mode)
        if self.difficulty is not None:
            if self.difficulty not in ale.getAvailableDifficulties():
                raise ValueError(
                    f"Invalid difficulty {self.difficulty}. Available difficulties: {ale.getAvailableDifficulties()}."
                )
            ale.setDifficulty(self.difficulty)

        action_set = ale.getMinimalActionSet() if self.action_set == ActionSet.Minimal else ale.getLegalActionSet()

        return ale, action_set


class AtariEnvironmentState(TypedDict):
    config: AtariEnvironmentConfig
    ale: ale_py.ALEState


class AtariEnvironmentObservation(TypedDict):
    rgb: NotRequired[npt.NDArray[np.uint8]]
    grayscale: NotRequired[npt.NDArray[np.uint8]]
    ram: NotRequired[npt.NDArray[np.uint8]]
    annotations: NotRequired[Mapping[str, int | Sequence[int]]]
    lives: int


class AtariEnvironmentObservationSpec(TypedDict):
    rgb: NotRequired[specs.Array]
    grayscale: NotRequired[specs.Array]
    ram: NotRequired[specs.Array]
    annotations: NotRequired[Mapping[str, specs.Array]]
    lives: specs.Array


class AtariEnvironment(AutoResetEnvironment):
    def __init__(
        self,
        game: AtariGame,
        mode: int | None = None,
        difficulty: int | None = None,
        seed: int | Sequence[int] | np.random.SeedSequence | None = None,
        repeat_action_probability: float = 0.25,
        frameskip: int = 1,
        max_episode_frames: int = 108_000,
        render_mode: RenderMode | None = None,
        frame_processing: FrameProcessing | None = None,
        action_set: ActionSet = ActionSet.Minimal,
        observation_type: ObservationType | Sequence[ObservationType] = (
            ObservationType.ImageRGB,
            ObservationType.Lives,
        ),
    ) -> None:
        super().__init__()
        self._config = AtariEnvironmentConfig(
            game=game,
            mode=mode,
            difficulty=difficulty,
            seed=seed,
            repeat_action_probability=repeat_action_probability,
            frameskip=frameskip,
            max_episode_frames=max_episode_frames,
            render_mode=render_mode,
            frame_processing=frame_processing,
            action_set=action_set,
            observation_type=observation_type,
        )
        self._ale, self._action_set = self._config.interface()
        self._metadata = getattr(game_metadata, self._config.game)

        if isinstance(observation_type, ObservationType):
            observation_type = [observation_type]

        self._observation_types = {obs_type.value: obs_type for obs_type in observation_type}

    @property
    def ale(self) -> ale_py.ALEInterface:
        return self._ale

    @property
    def config(self) -> AtariEnvironmentConfig:
        return self._config

    def _observation(self) -> AtariEnvironmentObservation:
        def _map_observation(
            observation_type: ObservationType,
        ) -> npt.NDArray[np.uint8] | int | dict[str, npt.NDArray[np.uint8] | int]:
            if observation_type == ObservationType.ImageRGB:
                return self._ale.getScreenRGB()
            elif observation_type == ObservationType.ImageGrayscale:
                return self._ale.getScreenGrayscale()
            elif observation_type == ObservationType.RAM:
                return self._ale.getRAM()
            elif observation_type == ObservationType.Lives:
                return self._ale.lives()
            elif observation_type == ObservationType.Annotations:
                ram = self._ale.getRAM()
                return {key: ram[value] for key, value in self._metadata.annotations.items()}
            else:
                raise ValueError(f"Invalid observation type {observation_type}")

        return jax.tree.map(_map_observation, self._observation_types)

    def reward_spec(self) -> specs.Array:
        return specs.Array(shape=(), dtype=np.int32, name="reward")

    def observation_spec(self) -> AtariEnvironmentObservationSpec:
        def _map_observation_spec(
            observation_type,
        ) -> specs.Array | dict[str, specs.Array]:
            height, width = self._ale.getScreenDims()
            if observation_type == ObservationType.ImageRGB:
                shape = (height, width, 3)
            elif observation_type == ObservationType.ImageGrayscale:
                shape = (height, width)
            elif observation_type == ObservationType.RAM:
                shape = (128,)
            elif observation_type == ObservationType.Lives:
                return specs.Array(shape=(), dtype=np.int32, name="lives")
            elif observation_type == ObservationType.Annotations:
                annotations = self._metadata.annotations
                return {
                    key: specs.Array(
                        shape=() if isinstance(value, int) else (len(value),),
                        dtype=np.uint8,
                        name=key,
                    )
                    for key, value in annotations.items()
                }
            else:
                raise ValueError(f"Unknown observation type: {observation_type}")
            return specs.Array(
                shape=shape,
                dtype=np.uint8,
                name=observation_type.name,
            )

        return jax.tree.map(_map_observation_spec, self._observation_types)

    def action_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(
            num_values=len(self._action_set),
            name="action",
        )

    def discount_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=(),
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
            name="discount",
        )

    def _step(self, action: int | ale_py.Action) -> dm_env.TimeStep:
        if isinstance(action, int):
            action = self._action_set[action]

        reward = self._ale.act(action)
        observation = self._observation()
        is_terminal = self._ale.game_over(with_truncation=False)
        is_truncated = self._ale.game_truncated()

        if is_terminal:
            return dm_env.termination(reward=reward, observation=observation)
        elif is_truncated:
            return dm_env.truncation(reward=reward, observation=observation)
        else:
            return dm_env.transition(reward=reward, observation=observation)

    @property
    def frame_count(self) -> int:
        return self.ale.getFrameNumber()

    @property
    def episode_frame_count(self) -> int:
        return self.ale.getEpisodeFrameNumber()

    def _reset(self) -> dm_env.TimeStep:
        self._ale.reset_game()
        observation = self._observation()
        return dm_env.restart(observation=observation)

    def __repr__(self) -> str:
        return f"AtariEnvironment{{ " f"game={self.config.game}, " f"frames={self.frame_count} }}"

    def __str__(self) -> str:
        return self.__repr__()

    def __getstate__(self) -> AtariEnvironmentState:
        return AtariEnvironmentState(
            config=self._config,
            ale=self._ale.getSystemState(),  # type: ignore
        )

    def __setstate__(self, state: AtariEnvironmentState) -> None:
        self._config = state["config"]
        self._ale, self._action_set = self._config.interface()
        self._ale.loadState(state["ale"])  # type: ignore
