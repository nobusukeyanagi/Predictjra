Predictjra v53.4 — Multi-source complete historical backfill

目的
- v53.3で多数発生した「安全性のため隔離した日程」を正常系から廃止する。
- 成功した全期間refreshでは、対象全race_idのresult+payoutが検証済みで skippedDates=0 を必須にする。

根本原因と修正
1. 旧: 観測した開催枠から1R〜12Rを機械的に生成
   新: SportsNaviのSSR開催一覧を主系統、netkeiba静的日別一覧を副系統として、実在race_idだけを列挙。
       短縮・中止開催で存在しないrace_idは生成しない。

2. 旧: Web補完が5レース連続失敗すると、以後の補完を全停止
   新: circuit breakerを廃止。全race_idを独立処理し、1件の失敗が後続レースへ波及しない。

3. 旧: payoutだけ欠けてもresult全体を再取得
   新: resultが正常ならpayoutだけ取得・検証・保存。正常なresultは変更しない。

4. 旧: 不完全日を隔離してAction成功
   新: 隔離成功を廃止。SportsNavi/netkeibaで補完後も未解決なら、race_id・不足側を明示してfail-closed。
       成功時は必ず「隔離日程: 0件」。

データ純度
- target raceの当日オッズ、実人気、馬体重、着順、時計、着差は予想入力用cardに保存しない。
- resultからcardを再構成する場合も、レース前に確定していた固定項目だけを投影する。
- 現行のprediction_logic_candidate.py / production.pyは変更しない。

適用後の最初の実行
Actions → Rebuild historical predictions
- scope: all
- mode: validate
- cache_policy: refresh

冒頭に以下が出ればv53.4が反映済み:
Backfill cache implementation: predictjra-historical-facts-v7-multisource-complete

成功時のActions Summary:
- 隔離日程: 0件（全対象日を完全性検証済み）
