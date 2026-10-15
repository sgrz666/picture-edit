set -u
cd /home/shangguanrz/project/pic-edit || exit 1
: > /tmp/readme_hand.txt
: > /tmp/readme_generated.txt
while IFS= read -r f; do
  if grep -q 'gen_folder_docs' "$f" 2>/dev/null; then echo "$f" >> /tmp/readme_generated.txt; else echo "$f" >> /tmp/readme_hand.txt; fi
done < <(find . -type f -name 'README*.md' -not -path './.git/*' | sed 's|^\./||')
echo "repo generated=$(wc -l < /tmp/readme_generated.txt) hand=$(wc -l < /tmp/readme_hand.txt)"
echo '@@@ HAND-WRITTEN PROTECT LIST (depth<=4 only) @@@'
awk -F/ 'NF<=4' /tmp/readme_hand.txt | head -80
echo '@@@ /data side classification @@@'
cd /data/shangguanrz || exit 1
: > /tmp/data_readme_hand.txt
while IFS= read -r f; do
  grep -q 'gen_folder_docs' "$f" 2>/dev/null || echo "${f#./}" >> /tmp/data_readme_hand.txt
done < <(find . -type f -name 'README*.md' 2>/dev/null | sed 's|^\./||')
echo "data hand=$(wc -l < /tmp/data_readme_hand.txt)"
awk -F/ 'NF<=4' /tmp/data_readme_hand.txt | head -40
