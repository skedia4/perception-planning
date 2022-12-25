#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@Time    :   2022/05/06 18:13:42
@Author  :   Yu Zhou 
@Contact :   yuzhou7@illinois.edu
'''

import yaml
import pprint

class ConfigSystem:
    def __init__(self, configFilename, verbose = False):
        self.verbose = verbose
        self.value = self.load(configFilename)

    def load(self, configFilename):
        with open(configFilename) as configFile:
            configDict = yaml.safe_load(configFile)
            if self.verbose:
                print(f"load config...")
                pprint.pprint(configDict)
        return configDict