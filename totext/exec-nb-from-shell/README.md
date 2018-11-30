# Test execute jupyter from shell

``` bash
 2018-11-30 16:34:02 ⌚  dell13 in ~/notebooks/totext/exec-nb-from-shell
± |master ✓| → jupyter nbconvert --to notebook --execute testprint.ipynb 
[NbConvertApp] WARNING | Config option `template_path` not recognized by `NotebookExporter`.
[NbConvertApp] Converting notebook testprint.ipynb to notebook
[NbConvertApp] Executing notebook with kernel: python3
[NbConvertApp] Writing 811 bytes to testprint.nbconvert.ipynb

 2018-11-30 16:34:41 ⌚  dell13 in ~/notebooks/totext/exec-nb-from-shell
± |master ✓| → ls
testprint.ipynb  testprint.nbconvert.ipynb
```
