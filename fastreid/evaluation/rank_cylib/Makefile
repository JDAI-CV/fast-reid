all:
	python3 setup.py build_ext --inplace
	rm -rf build
	python3 test_cython.py
clean:
	rm -rf build
	rm -f rank_cy.c *.so
