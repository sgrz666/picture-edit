// Offline browser QA and PDF export. Does not navigate to any network service.
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {createRequire} = require('node:module');
const {execFileSync} = require('node:child_process');
const packageRoot = process.env.CODEX_NODE_PACKAGES;
if (!packageRoot) throw Error('Set CODEX_NODE_PACKAGES to the bundled workspace package directory');
const bundledRequire = createRequire(path.join(packageRoot, 'package.json'));
const {chromium} = bundledRequire('playwright');

(async()=>{
  const [input, output] = process.argv.slice(2);
  if (!input || !output) throw Error('Usage: node render_qwen21_fixedpair_report.cjs report.html output-directory');
  if (fs.existsSync(output)) throw Error('Use a new immutable QA directory');
  fs.mkdirSync(output,{recursive:true});
  const browser = await chromium.launch({headless:true,
    ...(process.env.CODEX_REPORT_CHROMIUM?{executablePath:process.env.CODEX_REPORT_CHROMIUM}:{})});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1100},deviceScaleFactor:1});
    const errors=[];
    page.on('pageerror',e=>errors.push(String(e)));
    await page.route(/^https?:/,route=>route.abort());
    await page.goto(pathToFileURL(path.resolve(input)).href,{waitUntil:'load',timeout:120000});
    await page.waitForFunction(()=>window.REPORT_READY===true,{timeout:120000});
    await page.evaluate(()=>document.fonts.ready);
    const images=await page.locator('img[data-image]').evaluateAll(xs=>xs.map(x=>({ok:x.complete&&x.naturalWidth>0,alt:x.alt,sha:x.dataset.image})));
    if (images.some(x=>!x.ok)) throw Error('Missing report image');
    const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth);
    if(overflow)throw Error('Horizontal document overflow');
    // Verify the original-resolution viewer actually opens and closes.
    await page.locator('img[data-image]').first().click();
    await page.waitForFunction(()=>document.querySelector('dialog').open);
    await page.keyboard.press('Escape');
    await page.screenshot({path:path.join(output,'overview.png')});
    const sections=page.locator('main>section');
    for(let i=0;i<await sections.count();i++){
      await sections.nth(i).screenshot({path:path.join(output,`section_${String(i).padStart(2,'0')}.png`),timeout:60000});
    }
    // Preserve the screen report's expand/collapse behavior, expand only the PDF.
    const sectionCount=await sections.count();
    const partsRoot=path.join(path.dirname(input),'print_parts');
    if(fs.existsSync(partsRoot)){
      if(!process.env.CODEX_REPORT_PYTHON)throw Error('Set CODEX_REPORT_PYTHON to bundled Python for PDF merging');
      await page.close();
      const pdfs=[];
      for(const filename of fs.readdirSync(partsRoot).filter(n=>/^part_\d+\.html$/.test(n)).sort()){
        const part=await browser.newPage();
        await part.route(/^https?:/,route=>route.abort());
        await part.goto(pathToFileURL(path.resolve(partsRoot,filename)).href,{waitUntil:'load'});
        await part.evaluate(()=>document.fonts.ready);
        if(!await part.locator('img').evaluateAll(xs=>xs.every(x=>x.complete&&x.naturalWidth>0)))throw Error('Missing print image');
        await part.locator('details').evaluateAll(xs=>xs.forEach(x=>x.open=true));
        const pdf=path.resolve(output,filename.replace('.html','.pdf'));
        await part.pdf({path:pdf,printBackground:true,preferCSSPageSize:true,tagged:false,timeout:120000});
        pdfs.push(pdf);await part.close();
      }
      const merge='import sys; from pypdf import PdfWriter; w=PdfWriter(); [w.append(p) for p in sys.argv[2:]]; w.write(sys.argv[1]); w.close()';
      execFileSync(process.env.CODEX_REPORT_PYTHON,['-c',merge,path.resolve(output,'V5固定样本实验结果.pdf'),...pdfs]);
    }else{
      await page.locator('details').evaluateAll(xs=>xs.forEach(x=>x.open=true));
      await page.pdf({path:path.join(output,'V5固定样本实验结果.pdf'),printBackground:true,preferCSSPageSize:true,tagged:false,timeout:120000});
    }
    const report={input:path.resolve(input),figures:images.length,all_images_decoded:true,horizontal_overflow:false,
      viewer_open_and_close:true,sections:sectionCount,page_errors:errors,external_requests_allowed:false,
      print_method:fs.existsSync(partsRoot)?'original-file section PDFs then merged':'single-document'};
    if(errors.length)throw Error(JSON.stringify(errors));
    fs.writeFileSync(path.join(output,'browser_qa.json'),JSON.stringify(report,null,2));
    process.stdout.write(JSON.stringify(report));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
