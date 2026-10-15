set -u
cd /data/shangguanrz || exit 1
: > /tmp/data_prot.txt
while IFS= read -r f; do
  grep -q 'gen_folder_docs' "$f" 2>/dev/null || echo "${f#./}" >> /tmp/data_prot.txt
done < <(find . -type f -name 'README*.md' 2>/dev/null | sed 's|^\./||')
echo "data_protect=$(wc -l < /tmp/data_prot.txt)"
sort /tmp/data_prot.txt
