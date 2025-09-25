full:
	pip install build twine

build: libcml.so
	python -m build

libcml.so: cml.c cml.h
	gcc -shared -o libcml.so -lm -fPIC cml.c


