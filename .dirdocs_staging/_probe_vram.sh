cd /home/shangguanrz/project/pic-edit/vram_lab
for f in run.py worker.py common.py analysis.py acceptance.py report.py package.py contact_sheet.py; do echo "===== $f ($(wc -l < $f) lines)"; sed -n '1,18p' $f; done
echo ==TASKS
grep -n "def cmd_\|add_parser\|argparse" run.py | head -30
echo ==OUTPUTDIRS
grep -rn "ROOT\|out_dir\|output" analysis.py | grep -i "path\|dir" | head -10
