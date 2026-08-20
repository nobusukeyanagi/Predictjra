# Historical facts cache

`Rebuild historical predictions` が使用する過去事実データの永続キャッシュです。

- 対象レースは旧prediction CSVの有無ではなく、各日 `data/race_cards/` にある **12桁の正規JRAレースファイルを全件** 列挙します。
- その日の正規race cardすべてに確定result/payoutが揃った日だけ「完了日」として採用し、日単位の部分取り込みはしません。
- race cardは `win_odds` / `popularity` / `horse_weight` を物理削除して保存します。
- 旧prediction CSVが存在する場合は、上記禁止列に加え `score` / `predicted_rank` / `ml_*` など旧モデル出力も物理削除し、race cardと出走馬集合が一致するときだけrunner snapshotとして使います。
- 旧prediction CSVがないレース（例: 2026-07-25/26の一部）は、サニタイズ済みrace cardから `race_id` と `horse_number` だけのrunner snapshotを生成します。旧予想値は復元・再利用しません。
- 保存対象: サニタイズ済み番組/出走情報、runner snapshot、確定成績、三連単払戻、対象馬の過去成績。
- 保存しないもの: 対象レースの当日オッズ・当日実人気・馬体重/増減、旧モデル予想値、Predictjraの指数、本命・対抗・相手・危険馬、的中判定、回収率。
- `cache_policy=auto`: キャッシュ形式・元アーカイブbranch HEAD・対象期間を確認し、更新不要なら保存済みキャッシュを再利用します。
- `cache_policy=refresh`: 元アーカイブを再取得してキャッシュを強制作成し直します。

生成される `manifest.json` と `history-source.tar.gz`（大きい場合は分割ファイル）はGitHub Actionsが管理します。手動編集は不要です。
