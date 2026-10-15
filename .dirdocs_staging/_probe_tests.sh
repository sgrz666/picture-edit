cd /home/shangguanrz/project/pic-edit/tests
for f in test_*.py; do
  doc=$(python3 -c "import ast,sys; d=ast.parse(open('$f',encoding='utf-8').read()); doc=ast.get_docstring(d) or ''; print(' '.join(doc.split())[:150])" 2>/dev/null)
  firstline=$(head -1 "$f")
  n=$(grep -c "^def test\|^    def test" "$f")
  echo "$f | tests=$n | ${doc:-$firstline}"
done
