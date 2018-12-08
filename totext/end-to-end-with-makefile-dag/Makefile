all: demo-exec.md demo-exec.html
	$(MAKE) browser
	@echo "success"

demo-exec.html: demo-exec.ipynb
	jupyter nbconvert --to html $<

demo-exec.md: demo-exec.ipynb
	jupyter nbconvert --to markdown $<

demo-exec.ipynb: demo-edited.ipynb
	@echo "Executing final notebook draft end to end..."
	jupyter nbconvert --to notebook --execute $< --output $@

demo-edited.ipynb: demo-edited.py demo-editable.ipynb
	@echo "Copying the notebook so user can easily diff the results of this pipeline..."
	cp demo-editable.ipynb $@
	@echo "Taking user refactor from the .py file and updating the notebook..."
	jupytext --to notebook --update $< --output $@

demo-edited.py: demo-editable.py
	cp $< $@
	@echo "Please manually refactor the python code..."
	nano $@ || vi $@

demo-editable.py: demo-editable.ipynb
	@echo "Creating a .py from notebook for the user to refactor..."
	jupytext $< --to py:percent -o $@

demo-editable.ipynb:
	@echo "################"
	@echo "Creating a notebook for the user to edit..."
	@echo "Feel free to manually edit and run cells at this point, dev workflow..."
	cp demo-template.ipynb $@
	@echo "jupyter notebook $@"
	@echo "################"

clean:
	rm -rf demo-editable.ipynb demo-editable.py demo-edited.py demo-edited.ipynb demo-exec.ipynb demo-exec.md demo-exec.html

browser: demo-exec.html
	chromium $< || chrome-browser $< || firefox $< || xdg-open $<

install:
	pip install jupyter jupytext

install-conda:
	conda install jupyter
	conda install -c conda-forge jupytext
