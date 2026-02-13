#!/bin/bash

python3.10 -m venv venv

source venv/bin/activate
pip install -r requirements_h82_1.txt
pip install -r requirements_h82_2.txt --no-deps