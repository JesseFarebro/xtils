import asyncio
import concurrent.futures
import functools
import io
import logging
import tarfile
import typing
from typing import Optional

import pandas as pd
from aiohttp_client_cache import CachedSession, FileBackend  # pyright: ignore

from xtils.domains.atari import Atari57Game

logger = logging.getLogger(__name__)

AtariGames = set(typing.get_args(Atari57Game))


def _reindex_final_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sort the final dataframe by the mean score."""
    return df.reindex(columns=["Agent", "Game", "Seed", "Iteration", "Episodic Return"])


@functools.cache
def dopamine() -> pd.DataFrame:
    """Downloads Dopamine baseline results into a DataFrame."""

    async def fetch_game_data(
        session: CachedSession,
        game: Atari57Game,
    ) -> Optional[pd.DataFrame]:
        async with session.get(
            f"https://raw.githubusercontent.com/google/dopamine/master/baselines/atari/data/{game.lower()}.json"
        ) as response:
            if response.status != 200:
                logger.warning("Failed to fetch Dopamine data for %s", game)
                return None
            df = pd.read_json(io.StringIO(await response.text()))
            # Insert game name and seed number
            df["Game"] = game
            df["Seed"] = df.groupby(["Agent", "Game", "Iteration"]).cumcount()
            df = df.rename(
                columns={
                    "Value": "Episodic Return",
                }
            )
            return df

    async def fetch_all_games() -> pd.DataFrame:
        async with CachedSession(cache=FileBackend(use_temp=True)) as session:
            tasks = [fetch_game_data(session, game) for game in AtariGames]
            return pd.concat(filter(lambda df: df is not None, await asyncio.gather(*tasks)))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, fetch_all_games())
        df = typing.cast(pd.DataFrame, future.result())
    df = df.reset_index(drop=True)
    return _reindex_final_dataframe(df)


@functools.cache
def zoo() -> pd.DataFrame:
    """Downloads the DQN Zoo results into a DataFrame."""

    async def fetch_all_games() -> pd.DataFrame:
        dfs = []
        async with CachedSession(cache=FileBackend(use_temp=True)) as session:
            async with session.get("https://github.com/deepmind/dqn_zoo/raw/master/results.tar.gz") as response:
                with tarfile.open(fileobj=io.BytesIO(await response.read())) as tar:
                    for member in tar.getmembers():
                        if not member.name.endswith(".csv"):
                            continue
                        df = pd.read_csv(
                            tar.extractfile(member),  # pyright: ignore
                            usecols=[
                                "eval_episode_return",
                                "iteration",
                                "environment_name",
                                "seed",
                            ],
                        )
                        df = df.rename(
                            columns={
                                "eval_episode_return": "Episodic Return",
                                "iteration": "Iteration",
                                "environment_name": "Game",
                                "seed": "Seed",
                            }
                        )
                        df["Game"] = df.Game.apply(lambda game: game.replace("_", " ").title().replace(" ", ""))
                        df["Agent"] = member.name.removesuffix(".csv").title()
                        dfs.append(df)
        return pd.concat(dfs)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, fetch_all_games())
        df = typing.cast(pd.DataFrame, future.result())

    df = df.reset_index(drop=True)

    # Verify all games are there.
    for game in AtariGames - set(df.Game.unique()):
        logger.warning("Game %s doesn't exist in DQN Zoo data.", game)

    return _reindex_final_dataframe(df)
