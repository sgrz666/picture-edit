"""Use packaged rasterization with native Word on Windows without bundled LO."""
import importlib.util,os,pathlib,subprocess,sys
base=pathlib.Path(__file__).resolve().parent
skill=pathlib.Path('C:/Users/sg/.codex/plugins/cache/openai-primary-runtime/documents/26.1009.11318/skills/documents/render_docx.py')
spec=importlib.util.spec_from_file_location('packaged_docx_renderer',skill)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
poppler=pathlib.Path('C:/Users/sg/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin')
os.environ['PATH']=str(poppler)+os.pathsep+os.environ['PATH']
def convert_to_pdf(doc_path,user_profile,convert_tmp_dir,stem,verbose=False):
    target=pathlib.Path(convert_tmp_dir)/f'{stem}.pdf'
    result=subprocess.run(['powershell.exe','-NoProfile','-File',str(base/'render_word_native.ps1'),str(doc_path),str(target)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
    if result.returncode or not target.is_file():raise RuntimeError(result.stdout+'\n'+result.stderr)
    print(result.stdout.strip())
    return str(target),'Windows native Word read-only PDF export'
module.convert_to_pdf=convert_to_pdf
module.main()
