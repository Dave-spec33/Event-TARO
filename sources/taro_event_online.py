"""Thin trainer bridge for Event-TARO.

This intentionally preserves the same taro_generate_step(trainer, batch, gen_batch)
interface used by the already-working staged TARO patch in ray_trainer.py.
"""


def taro_generate_step(trainer, batch, gen_batch):
    del batch  # Event manager returns a self-contained variable-G PPO batch.

    gen_batch.meta_info["global_steps"] = trainer.global_steps
    output = trainer.async_rollout_manager.generate_sequences(gen_batch)

    taro_metrics = output.meta_info.pop("taro_metrics", {})
    # Keep the large event record on disk; do not forward it into trainer metrics.
    output.meta_info.pop("taro_event_record", None)
    output.meta_info.pop("timing", None)

    return output, taro_metrics
