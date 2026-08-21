# Historical facts cache

`Rebuild historical predictions` が使用する過去事実データの永続キャッシュです。

## 2026年1月以降のBackfill

- 2026年の確定result/payoutを実際の日付で列挙し、**2026-01-04（JRA最初の開催日）以降**を対象にします。
- 2026-07-18以降など、保存済みの正規race cardがある日はそのrace cardを優先します。
- race cardが保存されていない過去日は、確定resultから **レース前に確定していた固定項目だけ** を抽出してrace cardを再構成します。
  - 保存する: race_id、レース名、芝/ダート、距離、枠番、馬番、馬名、性齢、斤量、騎手、調教師、horse/jockey/trainer ID
  - 絶対に保存しない: 対象レースの着順、時計、着差、当日実人気、当日単勝オッズ、馬体重/増減
- 対象馬の過去走は2023〜2026年の確定resultから収集し、Rebuild側で **`date < target_date` の行だけ** を使用します。したがって1月の予想でも2025年以前の近走を使えますが、同日・未来の結果は使いません。

## Fail-closed 完全性検査

キャッシュ更新では誤った/部分的な履歴を採用しないことを優先します。次のどれか1つでも不整合があれば、更新を中止して既存キャッシュを残します。

- 各JRA開催日について1R〜12Rの結果がすべて揃っている
- 同一開催の開催日番号に欠番がない
- 同一race_id/開催枠が複数日付へ対応していない
- resultとpayoutのrace_idが一致する
- 出走した馬に確定人気が存在する
- 1〜3着が確定している
- 三連単の組合せが確定1〜3着と一致し、払戻額が正の値である
- 保存済みrace cardがある日はresult側のレース集合と完全一致する
- サニタイズ後のrace cardに当日オッズ・実人気・馬体重が残っていない

旧prediction CSVが存在する場合も、`score` / `predicted_rank` / `ml_*` など旧モデル出力と禁止列を物理削除し、race cardと出走馬集合が一致するときだけrunner snapshotとして使います。それ以外はサニタイズ済みcardから `race_id` / `horse_number` だけのsnapshotを生成します。

## Actionでの利用

このv4形式へ切り替える最初の1回は、GitHub Actionsの `Rebuild historical predictions` を次の設定で実行します。

1. `scope = all`
2. `mode = validate`
3. `cache_policy = refresh`

正常終了し、Actions Summaryの対象期間・レース数・Backfill開催日数を確認した後、採用するときだけ `all / apply / auto` を実行します。

`cache_policy=auto` はキャッシュ形式と元アーカイブbranch HEADを確認し、更新不要なら保存済みキャッシュを再利用します。`refresh` は元アーカイブを再取得してキャッシュを強制作成し直します。

生成される `manifest.json` と `history-source.tar.gz`（大きい場合は分割ファイル）はGitHub Actionsが管理します。手動編集は不要です。
