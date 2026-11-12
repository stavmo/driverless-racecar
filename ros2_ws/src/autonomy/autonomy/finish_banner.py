"""
Plain ASCII only
"""

WIDTH = 64


def _checkers(offset):
    row = ("##  " * (WIDTH // 4 + 1))[offset:offset + WIDTH]
    return row


def _center(text):
    return text.center(WIDTH)


def finish_banner(car_name, author, summary):
    laps = summary.get("lap_times_s", []) if summary else []
    cones = summary.get("cones_hit_total", "?") if summary else "?"
    off_track = summary.get("off_track_time_s", "?") if summary else "?"
    finished = len(laps) >= 2

    lines = [_checkers(0), _checkers(2), _checkers(0), ""]
    if finished:
        lines.append(_center("F I N I S H   -   M I S S I O N   C O M P L E T E"))
    else:
        lines.append(_center("DID NOT FINISH   -   M I S S I O N   N O T   C O M P L E T E D"))
    lines += ["", _center(f"Driver:  {car_name}"), ""]
    names = ["Lap 1  (explore + map)", "Lap 2  (fast lap)"]
    for name, t in zip(names, laps):
        lines.append(_center(f"{name:<24}{t:>8.2f} s"))
    lines.append(_center(f"{'Cones hit':<24}{str(cones):>10}"))
    lines.append(_center(f"{'Time off track':<24}{str(off_track) + ' s':>10}"))
    lines += ["", _center(f"Built by {author}"), _center("BGRacing Autonomous"), "",
              _checkers(0), _checkers(2), _checkers(0)]
    return "\n".join(lines)


if __name__ == "__main__":
    print(finish_banner("Lightning McQueen", "Stav Mordekhai",
                        {"lap_times_s": [283.36, 104.14], "cones_hit_total": 0, "off_track_time_s": 0.0}))
