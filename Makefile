build: libcml.so
	python -m build

full:
	pip install build twine

libcml.so: cml.c cml.h
	gcc -shared -o libcml.so -lm -fPIC cml.c


