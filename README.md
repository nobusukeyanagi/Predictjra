# JRA 自動予想・検証ページ

GitHub Pages + GitHub Actions で、netkeiba のJRA出馬表を取得し、予想と結果検証を日別に蓄積する静的サイトです。

## 現在の仕様

- 表示順: 新しい日付が上。日付内は競馬場 → レース番号順。
- 予想: 3連単2頭軸マルチ、軸2頭 + 相手5頭。
- 1点100円、30点、1レース3,000円。
- 馬番はJRAの枠色で表示。通常運用では出馬表から馬番と枠番を同時に取得・保存して表示。
- 前日15:00（Asia/Tokyo）: `shutuba_past.html` の出馬表から馬番を取得し、7頭をランダム選出。いったん作成した予想は再実行しても保持。
- 当日18:00（Asia/Tokyo）: netkeiba の `result.html?race_id=...` から着順と3連単払戻を取得し、的中判定・払戻・回収率を反映。
- 表の「三連単」は予想の的中・不的中にかかわらず、そのレースで実際に確定した3連単払戻を表示。
- 同着で3連単の的中組が複数ある場合にも対応し、購入対象に複数の的中券が含まれれば予想払戻を合算。

## GitHub Pages 公開

1. GitHubの **Settings → Pages → Build and deployment → Source** を **GitHub Actions** にします。
2. `main` ブランチへファイルをアップロード／コミットすると `Deploy GitHub Pages` が自動実行されます。
3. `Update race data` の完了後も `Deploy GitHub Pages` が自動実行され、更新データが公開ページへ反映されます。
4. 公開用Workflow内で `_site/.nojekyll` を自動生成するため、リポジトリ直下に `.nojekyll` ファイルを置く必要はありません。

## Discord通知

予想・結果の自動更新時は、ページのデプロイ成功後にDiscordへ通知します。通常のデザイン差分アップロードでは通知しません。

GitHubの **Settings → Secrets and variables → Actions → New repository secret** で以下を登録します。

- Name: `DISCORD_WEBHOOK_URL`
- Secret: 通知先DiscordチャンネルのWebhook URL

## 手動実行

**Actions → Update race data → Run workflow** から実行できます。

- `prepare`: 出馬表取得 + 予想作成
- `result`: 結果取得 + 的中判定
- `target_date`: `2026-08-16` のように指定。空欄なら、prepareは翌日、resultは当日を自動対象にします。

## 自動更新時刻

- 前日15:00（Asia/Tokyo）: 予想作成
- 当日18:00（Asia/Tokyo）: 結果・的中判定

## 補足

netkeiba 側のHTML構造が変更された場合は `scripts/update_races.py` のセレクタ修正が必要です。レース一覧は通常HTTPでrace_idが見つからない場合のみSelenium/Chromeへフォールバックします。
