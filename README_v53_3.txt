Predictjra v53.3 — resilient historical backfill repair

今回のログで確認した直接原因:
- race.netkeiba.com/top/race_list.html はブラウザではレース一覧を表示するが、
  GitHub Actions の requests 取得ではレースID本体が返らない（JavaScript後読み）。
- v53.2 はこのページを必須にしたため 2026-01-04 で0件判定して停止した。

v53.3の変更:
1. 動的race_listを必須依存から除外。
2. 元履歴に1件でも存在するJRA開催枠（race_id先頭10桁）から1R〜12Rを決定的に復元。
3. db.netkeiba.com のサーバー描画された静的日別一覧は補助的なクロスチェックだけに使用。
4. 欠損結果は、前回の検証済みキャッシュ → db.netkeiba静的結果ページ → race.netkeiba結果ページ の順で補完。
5. HTTPはリトライ。連続失敗時はサーキットブレーカーで無限に待たない。
6. 1レースを補完できなくてもAction全体を失敗させず、その開催日だけ隔離。
7. 隔離日を予想データへ混入させないため、正確性は維持。
8. 元結果CSV自体が不正（surface/distance欠落等）の場合も、Web結果で置換可能。
9. historical source の git clone も3回リトライ。
10. キャッシュ形式を v6 に更新し、旧v5を誤再利用しない。

適用後の実行:
- scope: all
- mode: validate
- cache_policy: refresh

Action冒頭に以下が出れば適用成功:
Backfill cache implementation: predictjra-historical-facts-v6-resilient-repair

検証済み:
- Python syntax compile: OK
- GitHub Actions YAML parse: OK
- candidate prediction logic tests: OK
- production prediction logic tests: OK
- live predict-engine contract tests: OK
- historical backfill leakage/completeness/resilience tests: OK

注意:
外部サイトやGitHub自体の全面障害まで「絶対にエラーゼロ」と保証することはできません。
ただし、今回までのような個別レース取得失敗・race_list仕様変更・一時的HTTP失敗は、
原則として全Actionを停止させない構成に変更しています。
