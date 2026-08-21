# Predictjra v53.1 Backfill repair

## 原因
GitHub Actions は `scripts/test_build_history_cache_backfill.py` を実行する設定でしたが、
GitHub 上にそのファイルが存在せず、Backfill開始前に停止していました。
また `scripts/build_history_cache.py` も旧 v3 のままでした。

## この差分で上書き・追加するファイル
- `.github/workflows/rebuild-history.yml`
- `scripts/build_history_cache.py`
- `scripts/test_build_history_cache_backfill.py`

## 適用後
Actions → Rebuild historical predictions を以下で再実行してください。

- scope: all
- mode: validate
- cache_policy: refresh

Action冒頭で
`Backfill cache implementation: predictjra-historical-facts-v4-result-backfill`
と表示されれば、v53.1の実装ファイルが正しく揃っています。
