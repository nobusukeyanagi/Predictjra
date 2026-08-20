# Historical facts cache

`Rebuild historical predictions` が使用する過去事実データの永続キャッシュです。

- 保存対象: レース前に確定していた出走・番組情報、確定成績、三連単払戻、対象馬の過去成績
- 旧prediction CSVは出走馬集合の確認用途だけに使います。`win_odds` / `popularity` / `horse_weight` と、`score` / `predicted_rank` / `ml_*` など旧モデル出力は、**キャッシュへ保存する前に物理的に削除**します。
- 保存しないもの: 当日オッズ、当日実人気、馬体重/増減、旧モデルの予想値、Predictjra の指数、本命・対抗・相手・危険馬、的中判定、回収率
- `cache_policy=auto`: キャッシュ形式・元アーカイブのbranch HEAD・対象期間を確認し、更新不要なら保存済みキャッシュを再利用します。
- `cache_policy=refresh`: 元アーカイブを再取得してキャッシュを強制作成し直します。

この方式により、旧スナップショットにレース後情報が混在していた日でも、その情報を予想入力から完全に除去できる場合は検証対象へ含められます。読み取り不能・必須列不足、またはcard/result/payout不足の日程は引き続き除外します。

生成される `manifest.json` と `history-source.tar.gz`（大きい場合は分割ファイル）は GitHub Actions が管理します。手動編集は不要です。
