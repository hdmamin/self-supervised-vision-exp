todo:
	ack -R '(# |""")TODO' {py,pylib,analysis,reports,docker,notes,services} || :

nb:
	cp notebooks/TEMPLATE.ipynb notebooks/nb000-untitled.ipynb

