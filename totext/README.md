# Example notebook conversions


Testing the display of different conversion formats on github
``` bash
jupyter nbconvert --to html test_poisson.ipynb 
jupyter nbconvert --to markdown test_poisson.ipynb 
jupyter nbconvert --to python test_poisson.ipynb 
jupyter nbconvert --to script --stdout > test_poisson.script.ipynb 
jupyter nbconvert --to pdf test_poisson.ipynb 
```
