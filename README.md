# Class introspection

Appendix material: code from the paper [available here](https://misc.ka.ge/honours.pdf).

## Installing

Create a virtual environment and install the dependencies:

```sh
$ virtualenv env_honours -p python3
$ source env_honours/bin/activate
(env_honours) $ pip install -r requirements.txt --verbose
```

To run the notebooks, you'll need [Jupyter](https://jupyter.org) installed. I recommend also
installing [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/). Next, you'll need
to install the virtual environment into Jupyter as a kernel:

```sh
(env_honours) $ pip install ipykernel
(env_honours) $ python -m ipykernel install --user --name=honours
```

## Layout

The initial explorations discussed in Chapter 4 can be found in the `initial-explorations/` directory.

The pipeline discussed in Chapter 5 can be found in the `pipeline/` directory.

The web application discussed in Chapter 5 can be found in the `pipeline/explorer/` directory.
