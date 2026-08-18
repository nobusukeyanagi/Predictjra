# JRA 自動予想・検証ページ

GitHub Pages + GitHub Actions で、netkeiba のJRA出馬表を取得し、予想と結果検証を日別に蓄積する静的サイトです。

## 現在の仕様

- 表示順: 新しい日付が上。日付内は競馬場 → レース番号順。
- 予想: 3連単2頭軸マルチ、軸2頭 + 相手5頭。
- 1点100円、30点、1レース3,000円。
- 馬番はJRAの枠色で表示。通常運用では出馬表から馬番と枠番を同時に取得・保存して表示。
- 前日15:00（Asia/Tokyo）: `shutuba_past.html` の出馬表から馬番を取得し、7頭をランダム選出。いったん作成した予想は再実行しても保持。
- 当日18:00（Asia/Tokyo）: netkeiba の結果ページから着順と3連単払戻を取得し、的中判定・払戻・回収率を反映。
- 同着で3連単の的中組が複数ある場合にも対応し、購入対象に複数の的中券が含まれれば払戻を合算。

## 初期データについて

`data/races.json` には 2026-08-16 と 2026-08-15 の36レースずつを収録しています。結果と3連単払戻は終了済みレースの初期データです。

初期2日分だけは、当時の出馬表をActionsで前日取得したわけではないため、**デザイン確認用の固定ランダム予想**です。次回以降は前日15時に取得した実際の出馬表に存在する馬番だけから予想します。

## GitHub Pages 公開手順

1. このフォルダの中身をGitHubリポジトリのルートへアップロードします。
2. GitHubの **Settings → Pages → Build and deployment → Source** を **GitHub Actions** にします。
3. `main` ブランチへ反映すると `Deploy GitHub Pages` が動き、サイトが公開されます。`Update race data` の完了後にも再デプロイされるため、自動取得したデータが公開ページへ反映されます。
4. Actionsが `data/races.json` を自動コミットするため、リポジトリ/組織の設定でGitHub Actionsによる書き込みが禁止されている場合は、**Settings → Actions → General → Workflow permissions** も確認してください。

## 手動実行

**Actions → Update race data → Run workflow** から実行できます。

- `prepare`: 出馬表取得 + 予想作成
- `result`: 結果取得 + 的中判定
- `target_date`: `2026-08-16` のように指定。空欄なら、prepareは翌日、resultは当日を自動対象にします。

## ファイル構成

```text
.
├─ index.html
├─ styles.css
├─ app.js
├─ data/
│  └─ races.json
├─ scripts/
│  └─ update_races.py
├─ .github/workflows/
│  ├─ update-races.yml
│  └─ pages.yml
├─ requirements.txt
└─ .nojekyll
```

## 補足

netkeiba 側のHTML構造が変更された場合は `scripts/update_races.py` のセレクタ修正が必要です。レース一覧はJavaScript描画に備えて、通常HTTPでrace_idが見つからない場合のみSelenium/Chromeへフォールバックします。
