#!/usr/bin/env python3
"""Regression tests for Predictjra live prediction rules."""
from __future__ import annotations

from predict_engine import MODEL_VERSION, build_prediction, parse_rich_card


def make_history(day, venue, finish, field, surf, dist, time_s, l3f, pos, margin):
    s = "ダ" if surf == "ダート" else "芝" if surf == "芝" else "障"
    return (
        f"{day} {venue} 3歳未勝利 {finish} {field}頭 1番 4人気 "
        f"{s}{dist} 良 {time_s} 騎手 57.0 480kg(0) "
        f"{l3f} {pos} 勝馬({margin})"
    )


def fixture_html(horse_count=14):
    rows = []
    for no in range(1, horse_count + 1):
        pasts = []
        for k in range(5):
            finish = ((no + k * 2) % 10) + 1
            pasts.append(make_history(
                f"2026.0{7-k}.1{(no+k)%9}",
                "札幌" if k % 2 == 0 else "函館",
                finish, 14, "ダート", 1700,
                f"1:4{7+k}.{no%10}",
                f"{36.0 + (no+k)%6/10:.1f}",
                f"{min(no,14)}-{max(1,min(no+k,14))}-{max(1,min(no+k+1,14))}",
                f"{max(0, finish-1)*0.2:.1f}",
            ))
        cells = "".join(f"<td class='Past'>{x}</td>" for x in pasts)
        rows.append(
            f"<tr><td class='Waku'>{1+(no-1)//2}</td>"
            f"<td class='Umaban'>{no}</td><td></td>"
            f"<td class='HorseName'><a href='/horse/2023{no:06d}/'>馬{no}</a></td>"
            f"<td>牡3 騎手57.0</td>{cells}</tr>"
        )
    return (
        "<html><body><h1>テスト重賞</h1>"
        "<div>13:00発走 / ダ1700m (右) / G2 / 馬場:良</div>"
        "<table class='Shutuba_Past5_Table'><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )


def main():
    base = [
        {"horse": n, "frame": 1+(n-1)//2, "name": f"馬{n}"}
        for n in range(1, 15)
    ]
    card = parse_rich_card(fixture_html(), "202601010901", base)
    built = build_prediction(card)

    assert MODEL_VERSION == "predictjra-live-index-v1"
    assert len(card["entries"]) == 14

    prediction = built["prediction"]
    assert len(prediction["axes"]) == 2
    assert len(prediction["opponents"]) == 5  # ceil(14/2)=7 selected
    assert len(prediction["excluded"]) == 1
    assert prediction["excluded"] == built["danger"]

    selected = set(prediction["axes"] + prediction["opponents"])
    assert not (selected & set(prediction["excluded"]))
    assert len(selected) == 7

    detail = built["indexDetail"]
    assert len(detail["horses"]) == 14
    assert all(len(h["recent"]) == 5 for h in detail["horses"])
    assert all("expectedPopularity" in h for h in detail["horses"])
    assert all("total" in h and "recentIndex" in h for h in detail["horses"])
    assert built["modelMeta"]["selectionRule"]

    # The current race's odds/popularity/bodyweight must never become model fields.
    serialized = str(built["modelMeta"]).lower()
    assert "prohibitedinputs" in serialized

    print("Predictjra live-rule regression tests: OK")


if __name__ == "__main__":
    main()
