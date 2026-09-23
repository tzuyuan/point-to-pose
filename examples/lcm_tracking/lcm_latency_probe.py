"""Measure LCM communication delay from a sensor publisher to this machine.

Subscribes to the rgbd channel (default from the pipeline config) and, for each
message, records:
    latency  = local receive time - msg.timestamp   (sensor stamp -> here)
    decode   = time to decode rgbd_t + unpack images
    gap      = inter-arrival time
No pipeline is involved, so this isolates the transport + subscriber thread.
Both processes must run on the same machine (or with synced clocks) for the
latency number to be meaningful.

    python examples/lcm_tracking/lcm_latency_probe.py --config configs/pipeline/lcm_tracking_crop.yaml
    python examples/lcm_tracking/lcm_latency_probe.py --channel d455_1 --report 1.0
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from point2pose.io.lcm.messages.rgbd_t import rgbd_t  # noqa: E402
from point2pose.io.lcm.image_conversion import unpack_image_from_bytes  # noqa: E402


def _stats(xs):
    a = np.asarray(xs, dtype=np.float64) * 1e3
    if a.size == 0:
        return "n=0"
    return "mean {:6.1f}  p50 {:6.1f}  p95 {:6.1f}  max {:6.1f} ms".format(
        a.mean(), np.percentile(a, 50), np.percentile(a, 95), a.max()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="pipeline yaml; reads lcm.rgbd_channel")
    ap.add_argument("--channel", type=str, default=None, help="override rgbd channel")
    ap.add_argument("--report", type=float, default=1.0, help="summary period, seconds")
    ap.add_argument("--per-msg", action="store_true", help="also print one line per message")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = run until Ctrl-C)")
    args = ap.parse_args()

    channel = args.channel
    if channel is None and args.config:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.config)
        channel = str(cfg.lcm.rgbd_channel)
    channel = channel or "d455_1"

    import lcm

    print(f"[probe] LCM_DEFAULT_URL={os.environ.get('LCM_DEFAULT_URL', '') or '(default udpm://239.255.76.67:7667)'}")
    print(f"[probe] subscribing to '{channel}'; summaries every {args.report:.1f}s. Ctrl-C to stop.")

    lc = lcm.LCM()
    window = {"lat": [], "dec": [], "gap": [], "bytes": 0, "n": 0}
    total = {"n": 0, "t0": time.time(), "last_rx": None, "lat_all": []}

    def handler(_ch, data):
        t_rx = time.time()
        msg = rgbd_t.decode(data)
        unpack_image_from_bytes(msg.rgb_image, height=msg.height, width=msg.width,
                                num_channels=msg.num_rgb_channels, channel_type=msg.rgb_channel_type)
        unpack_image_from_bytes(msg.depth_image, height=msg.height, width=msg.width,
                                num_channels=1, channel_type=msg.depth_channel_type)
        t_dec = time.time()
        lat = t_rx - float(msg.timestamp)
        window["lat"].append(lat)
        window["dec"].append(t_dec - t_rx)
        window["bytes"] += len(data)
        window["n"] += 1
        total["n"] += 1
        total["lat_all"].append(lat)
        if total["last_rx"] is not None:
            window["gap"].append(t_rx - total["last_rx"])
        total["last_rx"] = t_rx
        if args.per_msg:
            print(f"[probe] msg {total['n']:6d}  sensor->rx {1e3*lat:7.1f} ms  decode {1e3*(t_dec-t_rx):5.1f} ms  "
                  f"gap {1e3*window['gap'][-1] if window['gap'] else 0:6.1f} ms  size {len(data)/1e6:.2f} MB")

    lc.subscribe(channel, handler)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))

    next_report = time.time() + args.report
    while not stop["flag"]:
        lc.handle_timeout(50)
        now = time.time()
        if args.duration and now - total["t0"] > args.duration:
            break
        if now >= next_report:
            next_report = now + args.report
            n = window["n"]
            if n == 0:
                print(f"[probe] no messages on '{channel}' in the last {args.report:.1f}s "
                      f"(total {total['n']}); is the publisher running / same LCM URL?")
            else:
                hz = n / args.report
                mb = window["bytes"] / args.report / 1e6
                print(f"[probe] {hz:5.1f} Hz  {mb:5.1f} MB/s | sensor->rx {_stats(window['lat'])} | "
                      f"decode {_stats(window['dec'])} | gap {_stats(window['gap'])}")
            window.update(lat=[], dec=[], gap=[], bytes=0, n=0)

    el = time.time() - total["t0"]
    print(f"\n[probe] done: {total['n']} msgs in {el:.1f}s ({total['n']/max(el,1e-9):.1f} Hz); "
          f"sensor->rx overall {_stats(total['lat_all'])}")


if __name__ == "__main__":
    main()
