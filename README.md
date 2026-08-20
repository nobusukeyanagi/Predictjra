# Predictjra — JRA自動予想・検証ページ

GitHub Pages + GitHub Actions で、JRAの出走情報を取得し、予想・結果・回収率を日別に蓄積する静的サイトです。

## 現在の運用

- **13:00（Asia/Tokyo）**: 翌日の出走情報を取得し、**本番ロジック**で指数・本命・対抗・相手・危険馬を作成します。
- **19:00（Asia/Tokyo）**: 当日の結果と3連単払戻を取得し、的中判定・払戻・回収率を更新します。
- `Update race data` が正常終了すると、`Deploy GitHub Pages` が最新データを公開し、公開完了後にDiscordへ通知します。
- 表示順は新しい日付が上、同一日内は競馬場 → レース番号順です。

## 予想方式

- 券種は **3連単2頭軸マルチ**、1点100円です。
- 予想対象頭数は **出走頭数の2分の1切り上げ、最大7頭**です。
- 想定人気Top3のうち総合指数が最も低い馬を「危険馬」として予想対象から除外します。
- 残った馬から総合指数上位を予想対象とし、最上位を本命、選出馬のうち想定人気が最も低い馬を対抗、残りを相手とします。
- 相手は出走頭数に応じて1〜5頭となるため、購入点数は **6〜30点**、1レースの投資額は **600〜3,000円**です。
- 近5走の「展開・タイム・成績」、今回展開、コース・距離適性、総合指数、想定人気などを使用します。
- 現在レースの当日オッズ・実人気・馬体重/増減は予想入力に使用しません。過去レースの人気は、想定人気モデル用の過去情報としてのみ利用します。

## ロジックの候補／本番分離

予想ロジックは、同じ共通実装を **候補版と本番版の2つのスナップショット**として管理します。

- `scripts/prediction_logic_candidate.py` — **検証用候補**。通常のロジック調整ではこのファイルだけを変更します。
- `scripts/prediction_logic_production.py` — **本番採用版**。13:00の実戦予想は必ずこちらを使用します。
- `scripts/prediction_logic.py` — 旧importとの互換用。常に本番版を参照します。

`Rebuild historical predictions` の `validate` は候補版を使って過去検証しますが、本番版は変更しません。

`apply` を実行した場合だけ、Action内で候補版を本番版へ**そのままコピー**し、過去再計算がすべて成功した後に本番版と再計算結果を同じcommitで保存します。途中で失敗した場合、GitHub上の本番ロジックは変更されません。

このため、検証中の候補ロジックを `main` に置いたまま13:00を迎えても、`Update race data` は本番版を使い続けます。

## データ取得

通常運用では `scripts/update_races_v2.py` を実行します。

- レース発見・基本出走情報・結果取得は `scripts/update_races.py` の複数ソース対応処理を利用します。
- 近5走を含む詳細な出馬表は `scripts/predict_engine.py` で取得し、本番予想ロジックへ渡します。
- 取得元ごとの情報は可能な範囲で `dataSources` に記録します。

`update_races.py` 内には旧方式との互換用処理も残っていますが、**本番のprepare予想は `update_races_v2.py` → `predict_engine.py` → `prediction_logic_production.py` の経路で作成されます。**

## GitHub Actions

### Update race data

日々の自動運用です。

- 13:00: `prepare` — 翌日の出走情報・指数・予想を**本番ロジック**で作成
- 19:00: `result` — 当日の結果・的中判定・払戻・回収率を更新
- 正常終了後: GitHub Pagesへ公開し、その後Discord通知

このActionは候補ロジックを読みません。候補版が検証途中でも、日々の本番運用には影響しません。

手動実行では `target_date` を `YYYY-MM-DD` 形式で指定できます。空欄の場合、`prepare` は翌日、`result` は当日が対象です。

### Rebuild historical predictions

候補ロジックの変更検証・本番昇格・過去データ再計算に使用します。

通常の検証は次の設定です。

- `scope = all`
- `mode = validate`
- `cache_policy = auto`

`validate` は `prediction_logic_candidate.py` で過去レースを再計算し、成績を確認するだけです。`prediction_logic_production.py`、`data/races.json`、本番ページは変更しません。

採用するときだけ `mode = apply` を実行します。`apply` では、候補版を本番版へ昇格させ、同じロジックで過去データを再計算してcommitします。本番データが変わった場合はGitHub Pagesも再公開されます。Rebuildによる公開ではDiscord通知を送りません。

過去の出走情報・結果などの事実データは `data/history_cache/` に保存し、通常は `cache_policy = auto` で再利用します。旧prediction CSVに当日オッズ・実人気・馬体重や旧モデルの `score` / `predicted_rank` / `ml_*` が残っていても、それらをキャッシュ前に物理削除できる日程は検証対象に含めます。削除後のCSVは出走馬集合の確認だけに使い、旧予想値は再利用しません。`refresh` はキャッシュを元アーカイブから強制作成し直したい場合に使用します。

### Deploy GitHub Pages

公開専用のActionです。

- `Update race data` 成功後: **公開する + Discord通知**
- Rebuild `validate`: **公開しない**
- Rebuild `apply` で本番データ変更: **公開する**
- ロジック・テスト・スクリプトだけのpush: **公開しない**
- `index.html` / `app.js` / `styles.css` など公開UIの変更: **即時公開する**
- 手動再公開: `workflow_dispatch` から実行可能

デザイン変更では `data/races.json` を再計算しないため、既存の予想・結果・払戻・回収率はそのままです。

## Discord通知

Discord通知は、通常の `Update race data` によるページ公開が成功した後だけ送信します。

通知先はGitHubの **Settings → Secrets and variables → Actions** に `DISCORD_WEBHOOK_URL` を登録します。デザイン変更、Rebuild、手動DeployではDiscord通知を送りません。

## ロジック変更の基本フロー

1. ChatGPTへ変更方針を伝える
2. `scripts/prediction_logic_candidate.py` を中心とした差分をGitHubへアップロードする
3. `Rebuild historical predictions` を **`all / validate / auto`** で手動実行する
4. Actions Summaryで回収率・的中数・日別成績などを確認する
5. **不採用なら何もしない**。本番ロジックは最初から変更されていません
6. **採用する場合だけ** `all / apply / auto` を実行する
7. `apply` 成功後、候補ロジックが本番へ昇格し、次回の13:00更新から新ロジックが使われます

不採用の候補ファイルがGitHubに残っていても、本番運用には影響しません。次のロジック案を作るときは、現行の `prediction_logic_production.py` を基準に新しい候補版を作り直します。

## 補足

外部サイトのHTML構造が変更された場合は、取得処理のセレクタやフォールバック処理の修正が必要になることがあります。公開ページのデザインファイルは `Update race data` から変更されないようAction側で保護しています。
