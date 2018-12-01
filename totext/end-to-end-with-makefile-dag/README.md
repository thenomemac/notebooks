# End to End Jupyter workflow

This is an example of an end to end example of using jupyter notebooks in a production ready workflow with a DAG. In this case I use Makefiles as the dag engine.

The workflow goes as follows:

- Create a new editable notebook
- Edit said notebook (optional: open demo-editable.ipynb)
- Convert notebook to a python script and edit in IDE of your choice
- Update the notebook with the changes to the script
- Run the notebook end to end
- Export the final notebook as markdown for code review and searchability
- Export the notebook as html for portability
- Final step (not implemented): save/move the .py .ipynb .md .html artifacts
