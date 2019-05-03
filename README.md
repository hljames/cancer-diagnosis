# cancer-diagnosis

Steps to add a page:
jupyter nbconvert --output-dir . --to markdown --template ../gpages/_support/markdown.tpl notebooks/<notebook name>.ipynb

python ../gpages/_support/nbmd.py <notebook name>.md
  
Edit the markdown files to add a YAML tag nav_include: <position on nav bar>
