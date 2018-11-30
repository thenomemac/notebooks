# Jupytext Experiments

Overall impressions from experiment.
Jupytext is awesome!

Log of my iniial experiments with jupytext lib:

``` bash
 2018-11-30 15:50:44 ⌚  dell13 in ~/notebooks/totext/jupytext                                          
± |master ✓| → cp ../test_poisson.ipynb notebook.ipynb                                                  

 2018-11-30 15:51:10 ⌚  dell13 in ~/notebooks/totext/jupytext                                          
± |master ✓| → ls
notebook.ipynb

 2018-11-30 15:51:11 ⌚  dell13 in ~/notebooks/totext/jupytext                                          
± |master ✓| → jupyter nbconvert --to html test_poisson.ipynb ^C                                        

 2018-11-30 15:52:08 ⌚  dell13 in ~/notebooks/totext/jupytext
± |master ✓| → jupytext --to python -o notebook.py
jupytext: error: Please specificy either --from, --pre-commit or notebooks

 2018-11-30 15:52:43 ⌚  dell13 in ~/notebooks/totext/jupytext
± |master ✓| → jupytext --to python notebook.ipynb -o notebook.py
[jupytext] Converting 'notebook.ipynb' to 'notebook.py'

 2018-11-30 15:52:50 ⌚  dell13 in ~/notebooks/totext/jupytext
± |master ?:1 ✗| → jupytext notebook.ipynb --to py:percent -o notebook.percent.py
[jupytext] Converting 'notebook.ipynb' to 'notebook.percent.py' using format 'percent'

 2018-11-30 15:53:10 ⌚  dell13 in ~/notebooks/totext/jupytext
± |master ?:1 ✗| → jupytext notebook.ipynb --to py:light -o notebook.light.py
[jupytext] Converting 'notebook.ipynb' to 'notebook.light.py' using format 'light'

 2018-11-30 15:53:23 ⌚  dell13 in ~/notebooks/totext/jupytext
± |master ?:1 ✗| → jupytext notebook.ipynb --to py:light^[[5~^Co notebook.light.py

 2018-11-30 15:53:31 ⌚  dell13 in ~/notebooks/totext/jupytext
± |master ?:1 ✗| → jupytext notebook.ipynb --to markdown -o notebook.md
[jupytext] Converting 'notebook.ipynb' to 'notebook.md'

 2018-11-30 15:53:57 ⌚  dell13 in ~/notebooks/totext/jupytext
± |master ?:1 ✗| → jupytext notebook. --to notebook -o notebook.light.ipynb
notebook.ipynb       notebook.md          notebook.py          
notebook.light.py    notebook.percent.py  

 2018-11-30 15:53:57 ⌚  dell13 in ~/notebooks/totext/jupytext
± |master ?:1 ✗| → jupytext notebook.light.py --to notebook -o notebook.light.ipynb
[jupytext] Converting 'notebook.light.py' to 'notebook.light.ipynb'

 2018-11-30 16:00:46 ⌚  dell13 in ~/notebooks/totext/jupytext
± |master ?:1 ✗| → jupytext notebook.percent.py --to notebook -o notebook.percent.ipynb
[jupytext] Converting 'notebook.percent.py' to 'notebook.percent.ipynb'

 2018-11-30 16:01:16 ⌚  dell13 in ~/notebooks/totext/jupytext
± |master ?:1 ✗| → cp notebook.ipynb notebook.2.ipynb

 2018-11-30 16:05:13 ⌚  dell13 in ~/notebooks/totext/jupytext
± |master ?:1 ✗| → jupytext notebook.percent.2.py --to notebook --update -o notebook.2.ipynb
[jupytext] Converting 'notebook.percent.2.py' to 'notebook.2.ipynb' (destination file updated)

 2018-11-30 16:15:36 ⌚  dell13 in ~/notebooks/totext/jupytext
± |master ?:1 ✗| → cp notebook.ipynb notebook.3.ipynb; jupytext notebook.3.md --to notebook --update -o notebook.3.ipynb
[jupytext] Converting 'notebook.3.md' to 'notebook.3.ipynb' (destination file updated)


```
