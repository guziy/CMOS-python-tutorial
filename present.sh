#!/bin/bash

ipython nbconvert Python_CNRCWP_tutorial.ipynb --to slides --post serve --config slides_config.py --template default_transition.tpl
