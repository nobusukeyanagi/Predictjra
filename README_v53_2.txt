Predictjra v53.2 — 2026履歴Backfill 欠損補完修正

【今回のエラー原因】
v53.1 は外部履歴リポジトリを完全な結果アーカイブと仮定し、
各meeting-dayの1R〜12Rが揃っていることを要求していました。
実際の外部履歴には2026年の一部race_id/result/payoutが欠けていたため、
安全装置が Historical result archive has structural holes で停止しました。

【v53.2の変更】
1. netkeiba月間開催カレンダーから実際のJRA開催日を独立取得
2. 各開催日のレース一覧から実際の12桁race_id集合を取得
3. 外部履歴に欠けるresult/payoutだけnetkeiba確定結果ページから補完
4. 補完結果はresult上位3頭と三連単払戻を相互検証
5. 補完後のresult race_id集合が、独立取得した当日race_id集合と完全一致することを検証
6. 既存の正常なresult/payoutは上書きしない
7. 既存データとWeb補完結果が矛盾した場合はfail-closedで停止
8. 予想入力へは従来どおり着順・時計・当日人気・当日オッズ・馬体重を渡さない

【差し替え・追加対象】
- .github/workflows/rebuild-history.yml
- scripts/build_history_cache.py
- scripts/test_build_history_cache_backfill.py

【適用後の実行】
Actions → Rebuild historical predictions
- scope: all
- mode: validate
- cache_policy: refresh

冒頭に以下が表示されればv53.2が反映されています。
Backfill cache implementation: predictjra-historical-facts-v5-web-repair

validate成功後に本番反映する場合:
- scope: all
- mode: apply
- cache_policy: auto
