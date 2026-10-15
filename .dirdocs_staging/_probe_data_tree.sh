set -u
cd /data/shangguanrz || exit 1
echo '=== TOP LEVEL ==='
ls -la | sed 's/^/  /'
echo '=== DEPTH1/2 DIRECTORY TREE (with direct child counts) ==='
find . -maxdepth 2 -mindepth 1 -type d -not -path './.*' | sed 's|^\./||' | sort
echo '=== PER-TOP summary: subdirs / files at depth1 / any build-marker files ==='
for T in */ ; do
  T="${T%/}"
  sd=$(find "$T" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  fl=$(find "$T" -mindepth 1 -maxdepth 1 -type f 2>/dev/null | wc -l)
  echo "  [$T] subdirs=$sd files=$fl"
done
echo '=== HUMAN-WRITTEN DOCS (no generator marker, modified since 2026-10-12) ==='
find . -maxdepth 6 -type f \( -name 'README*.md' -o -name '目录*.md' \) -newermt '2026-10-12 00:00' 2>/dev/null | head -40
