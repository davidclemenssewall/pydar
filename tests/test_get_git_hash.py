#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_get_git_hash.py

Test our function for getting the current git hash of pydar

Created on Sat Mar 13 13:13:04 2021

@author: thayer
"""

import os
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

os.chdir('/home/thayer/Desktop/')

print(os.getcwd())

print(pydar.get_git_hash())

print(os.getcwd())