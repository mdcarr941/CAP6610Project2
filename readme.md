Dependencies
============

The code in this project is written for Python 2.7. The following libraries are
required: sklearn, numpy, and GPy. All of these are available in the PyPI and so
can be installed easily using pip.


Usage
=====

Simply run
```python
    import group12_code
```
to import all the code for the project. The calling conventions of each function are
documented in the functions themselves and are available from the interpreter by
calling
```python
    help(group12_code.<function name>)
```
Tests for the project are located in
`testing_script.py`. Execute this script to run all test, or import it into the
interpreter to run individual test cases. 

A dictionary argument is to be passed to to MyCrossValidate(X_Train, Nf, Parameters, y_Target) which contains the key,value pair with key 'algorithm' and can have the values 'RVM', 'SVM' or 'GPR'.
