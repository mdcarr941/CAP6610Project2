Dependencies
============

The code in this project is written for Python 2.7. The following libraries are
required: sklearn, numpy, and GPy. All of these are available in the PyPI and so
can be installed easily via pip using:
```python
    pip install numpy
    pip install sklearn
    pip install GPy
```


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
It is expected that the Parameters argument to MyCrossValidate and to
TrainMyClassifier will contain the key 'algorithm' which can point to the
values 'RVM', 'SVM' or 'GPR'. This value determines which classification
algorithm will be run
