#!/usr/bin/env python3
from __future__ import annotations

from predict_engine import MODEL_VERSION, build_prediction, parse_rich_card, parse_class_level


def make_history(day, venue, finish, field, popularity, class_name, carried, dist=2000):
    margin = max(0, finish - 1) * 0.2
    return (
        f"{day} {venue} {class_name} {finish} {field}頭 1番 {popularity}人気 "
        f"芝{dist} 良 1:59.0 酒井学 {carried:.1f} 450kg(0) "
        f"35.0 5-5-5-5 勝馬({margin:.1f})"
    )


def fixture_html(horse_count=14, zendan_like=False):
    rows = []
    for no in range(1, horse_count + 1):
        if zendan_like and no == 12:
            # Ability looks strong, but market memory is weak; most recent win was
            # a low-popularity handicap win at 52kg before moving to 58kg.
            histories = [
                make_history("2026.07.19", "小倉", 1, 18, 10, "小倉記念 GIII", 52.0),
                make_history("2026.06.14", "阪神", 5, 14, 11, "花のみちS 3勝", 58.0, 1600),
                make_history("2026.05.17", "京都", 5, 14, 7, "錦S 3勝", 55.0, 1600),
                make_history("2026.04.19", "阪神", 5, 14, 9, "立雲峡S 3勝", 58.0, 1600),
                make_history("2026.02.22", "阪神", 1, 12, 6, "天神橋特別 2勝", 55.0, 1600),
            ]
        else:
            # Market-fancied horses have a much stronger popularity memory.
            base_pop = 1 + ((no - 1) % 5)
            histories = [
                make_history(f"2026.0{7-k}.1{(no+k)%9}", "札幌", 2 + (k % 3), 14,
                             min(base_pop + (k % 2), 6), "重賞 GII", 57.0)
                for k in range(5)
            ]
        cells = "".join(f"<td class='Past'>{x}</td>" for x in histories)
        rows.append(
            f"<tr><td class='Waku'>{1+(no-1)//2}</td>"
            f"<td class='Umaban'>{no}</td><td></td>"
            f"<td class='HorseName'><a href='/horse/2023{no:06d}/'>馬{no}</a>"
            f"<a href='/trainer/result/recent/{1000+no:05d}/'>厩舎{no}</a></td>"
            f"<td>牡4 <a href='/jockey/result/recent/{2000+no:05d}/'>騎手{no}</a> "
            f"{58.0 if (zendan_like and no == 12) else 57.0:.1f}</td>{cells}</tr>"
        )
    return (
        "<html><body><h1>札幌記念</h1>"
        "<div>15:45発走 / 芝2000m (右) / G2 / オープン / 馬場:良</div>"
        "<table class='Shutuba_Past5_Table'><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )


def main():
    assert MODEL_VERSION == "predictjra-live-index-v2-market-memory"
    assert parse_class_level("小倉記念 GIII") == 5
    assert parse_class_level("札幌記念 GII") == 6
    assert parse_class_level("大阪杯 GI") == 7

    base = [{"horse": n, "frame": 1+(n-1)//2, "name": f"馬{n}"} for n in range(1, 15)]
    card = parse_rich_card(fixture_html(), "202601010901", base)
    built = build_prediction(card)
    assert built["modelMeta"]["logicSource"] == "scripts/prediction_logic.py"
    prediction = built["prediction"]
    assert len(prediction["axes"]) == 2
    assert len(prediction["opponents"]) == 5
    assert len(prediction["excluded"]) == 1
    assert len(built["indexDetail"]["horses"]) == 14
    assert all(len(h["recent"]) == 5 for h in built["indexDetail"]["horses"])

    market_card = parse_rich_card(fixture_html(16, zendan_like=True), "202601010811",
                                  [{"horse": n, "frame": 1+(n-1)//2, "name": f"馬{n}"} for n in range(1,17)])
    market_built = build_prediction(market_card)
    zendan = next(h for h in market_built["indexDetail"]["horses"] if h["no"] == 12)
    assert zendan["popularityContext"]["assignedWeightDelta"] == 6.0
    assert zendan["popularityFactors"]["last_lowpop_win"] == 100.0
    assert zendan["popularityFactors"]["handicap_rebound_risk"] >= 99.0
    # The point of v2 is to stop an ability-only model from making this horse a top-3 favorite.
    assert zendan["expectedPopularity"] > 3, zendan

    print("Predictjra market-memory live tests: OK")
    print("synthetic Zendan-like expected popularity:", zendan["expectedPopularity"])


if __name__ == "__main__":
    main()
