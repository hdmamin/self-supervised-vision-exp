todo:
	ack -R '(# |""")TODO' {bin,lib,notebooks,reports,notes,services} || :

nb:
	cp notebooks/TEMPLATE.ipynb notebooks/nb000-untitled.ipynb

