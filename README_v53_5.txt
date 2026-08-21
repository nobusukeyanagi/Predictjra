Predictjra v53.5 - JRA official executed-race backfill

原因:
- v53.4が開催予定のrace_idを「実施済み」と誤認。
- 2026-02-07 東京8R以降5レース、2026-02-08 小倉4Rは実際には降雪で取りやめ。
- result/payoutが存在しないのが正常なのに補完失敗として停止した。

修正:
- JRA公式の更新済み開催日程ページを実施レース集合の最優先ソースに変更。
- 「第Nレース取りやめ」「第Nレース以降取りやめ」「開催中止」を正常な非施行として除外。
- SportsNavi/netkeibaは結果・払戻補完のフォールバックとして維持。
- 両結果ソースに結果がなくても、JRA公式またはSportsNavi出馬表で中止確認できればエラーにしない。
- 中止確認できない不明な欠損だけはfail-closedで停止。
- cache version v8。

適用後:
Actions > Rebuild historical predictions
scope=all / mode=validate / cache_policy=refresh
