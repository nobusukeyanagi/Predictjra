# Historical facts cache

`Rebuild historical predictions` が使用する過去事実データの永続キャッシュです。

- 保存対象: 安全性確認済みのレース前出走情報、事前予想スナップショット、確定成績、三連単払戻、対象馬の過去成績
- 保存しないもの: Predictjra の指数、本命・対抗・相手・危険馬、的中判定、回収率
- `cache_policy=auto`: 元アーカイブの branch HEAD が同じ間は保存済みキャッシュを再利用します。
- `cache_policy=refresh`: 元アーカイブを再取得してキャッシュを作り直します。

生成される `manifest.json` と `history-source.tar.gz`（大きい場合は分割ファイル）は GitHub Actions が管理します。手動編集は不要です。
