# Predictjra — JRA自動予想・検証ページ

GitHub Pages + GitHub Actions で、JRAの出走情報を取得し、予想・結果・回収率を日別に蓄積する静的サイトです。

## 現在の運用

- **13:00（Asia/Tokyo）**: 翌日の出走情報を取得し、その時点で `main` にある予想ロジックで指数・本命・対抗・相手・危険馬を作成します。
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

予想ルールの正本は **`scripts/prediction_logic.py`** です。実戦予想の `scripts/predict_engine.py` と過去検証の `scripts/rebuild_history.py` は、同じ共通ロジックを使用します。

## データ取得

通常運用では `scripts/update_races_v2.py` を実行します。

- レース発見・基本出走情報・結果取得は `scripts/update_races.py` の複数ソース対応処理を利用します。
- 近5走を含む詳細な出馬表は `scripts/predict_engine.py` で取得し、共通予想ロジックへ渡します。
- 取得元ごとの情報は可能な範囲で `dataSources` に記録します。

`update_races.py` 内には旧方式との互換用処理も残っていますが、**本番のprepare予想は `update_races_v2.py` → `predict_engine.py` → `prediction_logic.py` の経路で作成されます。**

## GitHub Actions

### Update race data

日々の自動運用です。

- 13:00: `prepare` — 翌日の出走情報・指数・予想を作成
- 19:00: `result` — 当日の結果・的中判定・払戻・回収率を更新
- 正常終了後: GitHub Pagesへ公開し、その後Discord通知

手動実行では `target_date` を `YYYY-MM-DD` 形式で指定できます。空欄の場合、`prepare` は翌日、`result` は当日が対象です。

### Rebuild historical predictions

予想ロジックの変更検証・過去データ再計算に使用します。

通常のロジック検証は次の設定です。

- `scope = all`
- `mode = validate`
- `cache_policy = auto`

`validate` は現在のロジックで過去レースを再計算して成績を確認するだけで、`data/races.json` の本番予想・結果は上書きしません。

検証したロジックを過去データにも正式反映するときだけ、`mode = apply` を使用します。`apply` で本番データが変わった場合はGitHub Pagesも再公開されます。Rebuildによる公開ではDiscord通知を送りません。

過去の出走情報・結果などの事実データは `data/history_cache/` に保存し、通常は `cache_policy = auto` で再利用します。`refresh` はキャッシュを元アーカイブから強制作成し直したい場合だけ使用します。

## Deploy GitHub Pages

公開専用のActionです。

- `Update race data` 成功後: **公開する**
- Rebuild `validate`: **公開しない**
- Rebuild `apply` で本番データ変更: **公開する**
- `prediction_logic.py` などロジック・テスト・スクリプトだけのpush: **公開しない**
- `index.html` / `app.js` / `styles.css` など公開UIの変更: **即時公開する**
- 手動再公開: `workflow_dispatch` から実行可能

デザイン変更では `data/races.json` を再計算しないため、既存の予想・結果・払戻・回収率はそのままです。

GitHub Pagesの **Settings → Pages → Build and deployment → Source** は **GitHub Actions** を使用します。公開Workflow内で `_site/.nojekyll` を生成するため、リポジトリ直下の `.nojekyll` は不要です。

## Discord通知

Discord通知は、通常の `Update race data` によるページ公開が成功した後だけ送信します。

通知先はGitHubの **Settings → Secrets and variables → Actions** に以下を登録します。

- Name: `DISCORD_WEBHOOK_URL`
- Secret: 通知先DiscordチャンネルのWebhook URL

デザイン変更、Rebuild、手動DeployではDiscord通知を送りません。

## ロジック変更の基本フロー

1. `scripts/prediction_logic.py` を中心にロジックを変更する
2. GitHubへアップロードする（ロジック変更だけではPagesは公開されない）
3. `Rebuild historical predictions` を `all / validate / auto` で実行する
4. Actions Summaryで回収率・的中数・日別成績などを確認する
5. 不採用ならロジックを戻す
6. 採用する場合は `all / apply / auto` を実行し、過去データにも正式反映する

なお、次回の `Update race data` は、その実行時点で `main` にあるロジックを使用します。検証中のロジックを `main` に置いたまま13:00を迎える場合は、そのロジックで翌日の予想が作成されます。

## 補足

外部サイトのHTML構造が変更された場合は、取得処理のセレクタやフォールバック処理の修正が必要になることがあります。公開ページのデザインファイルは `Update race data` から変更されないようAction側で保護しています。
