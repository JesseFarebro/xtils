from typing import Any, Mapping

import aim
import fancyflags as ff
import wandb
from absl import flags
from clu import metric_writers
from clu.metric_writers.interface import Array

AIM_FLAGS = ff.DEFINE_dict(
    "aim",
    repo=ff.String(None, "Repository directory."),
    experiment=ff.String("dsr", "Experiment name."),
    run_hash=ff.String(None, "Run hash."),
    log_system_params=ff.Boolean(True, "Log system parameters."),
)
WANDB_FLAGS = ff.DEFINE_dict(
    "wandb",
    save_code=ff.Boolean(False, "Save code."),
    id=ff.String(None, "Run ID."),
    tags=ff.StringList(None, "Tags."),
    name=ff.String(None, "Name."),
    group=ff.String(None, "Group."),
    mode=ff.Enum("online", ["online", "offline", "disabled"], "Mode."),
)
TENSORBOARD_FLAGS = ff.DEFINE_dict(
    "tensorboard",
    logdir=ff.String("logdir", "Log directory."),
)
METRIC_WRITER = flags.DEFINE_enum(
    "metric_writer",
    "aim",
    ["aim", "wandb", "tensorboard"],
    "Metric writer to use.",
)

MetricWriter = metric_writers.MetricWriter


class WanDBWriter(metric_writers.MetricWriter):
    def __init__(self, /, **kwargs):
        self.run: wandb.wandb_sdk.wandb_run.Run = wandb.init(**kwargs, resume=True)  # type: ignore

    def write_summaries(self, step: int, values: Mapping[str, Array], metadata: Mapping[str, Any] | None = None):
        for key, value in values.items():
            self.run.log({key: value, **(metadata or {})}, step=step)

    def write_images(self, step: int, images: Mapping[str, Array]):
        for key, value in images.items():
            self.run.log({key: [wandb.Image(value)]}, step=step)

    def write_scalars(self, step: int, scalars: Mapping[str, float]):
        for key, value in scalars.items():
            self.run.log({key: value}, step=step)

    def write_hparams(self, hparams: Mapping[str, Any]):
        self.run.config.update(hparams)

    def write_audios(self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
        del step, audios, sample_rate
        raise NotImplementedError

    def write_histograms(self, step: int, arrays: Mapping[str, Array], num_buckets: Mapping[str, int] | None = None):
        self.run.log(
            {
                key: wandb.Histogram(array, num_bins=num_buckets[key] if num_buckets else 64)  # type: ignore
                for key, array in arrays.items()
            },
            step=step,
        )

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        del step, videos
        raise NotImplementedError

    def write_texts(self, step: int, texts: Mapping[str, str]):
        del step, texts
        raise NotImplementedError

    def close(self):
        pass

    def flush(self):
        pass


class AimWriter(metric_writers.MetricWriter):
    def __init__(self, /, **kwargs):
        self.run = aim.Run(**kwargs)

    def write_summaries(self, step: int, values: Mapping[str, Array], metadata: Mapping[str, Any] | None = None):
        for key, value in values.items():
            self.run.track(value, name=key, step=step, context=metadata)  # type: ignore
        self.run.report_progress()

    def write_images(self, step: int, images: Mapping[str, Array]):
        for key, value in images.items():
            self.run.track(aim.Image(value), name=key, step=step)
        self.run.report_progress()

    def write_scalars(self, step: int, scalars: Mapping[str, float]):
        for key, value in scalars.items():
            self.run.track(value, name=key, step=step)
        self.run.report_progress()

    def write_hparams(self, hparams: Mapping[str, Any]):
        self.run["hparams"] = hparams

    def write_audios(self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
        del step, audios, sample_rate
        raise NotImplementedError

    def write_histograms(self, step: int, arrays: Mapping[str, Array], num_buckets: Mapping[str, int] | None = None):
        for key, array in arrays.items():
            dist = aim.Distribution(
                array,
                bin_count=num_buckets[key] if num_buckets else 64,
            )
            self.run.track(dist, name=key, step=step)

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        del step, videos
        raise NotImplementedError

    def write_texts(self, step: int, texts: Mapping[str, str]):
        del step, texts
        raise NotImplementedError

    def close(self):
        self.run.close()

    def flush(self):
        pass


def create_default_writer(
    *,
    just_logging: bool = False,
    asynchronous: bool = False,
) -> metric_writers.MultiWriter:
    """Create the default writer for the platform."""
    if just_logging:
        if asynchronous:
            return metric_writers.AsyncMultiWriter([metric_writers.LoggingWriter()])
        else:
            return metric_writers.MultiWriter([metric_writers.LoggingWriter()])
    writers: list[metric_writers.MetricWriter] = [metric_writers.LoggingWriter()]

    match METRIC_WRITER.value:
        case "wandb":
            writers.append(WanDBWriter(**WANDB_FLAGS.value))
        case "aim":
            writers.append(AimWriter(**AIM_FLAGS.value))
        case "tensorboard":
            writers.append(metric_writers.SummaryWriter(**TENSORBOARD_FLAGS.value))
        case _:
            raise ValueError(f"Unknown metric writer: {METRIC_WRITER.value}")

    if asynchronous:
        return metric_writers.AsyncMultiWriter(writers)
    return metric_writers.MultiWriter(writers)
