import numpy as np
import pandas as pd
import sys


def parse_zipcode():
    if len(sys.argv) <= 1:
        raise Exception("missing argument")

    valid_zips = ["1", "2", "3", "4"]

    zip_arg = sys.argv[1]

    arg_zip = zip_arg.split("=")
    if (arg_zip[0] != "-zipcode"):
        raise Exception("unknown flag: "+arg_zip[0])
    if (len(arg_zip) < 2):
        raise Exception("no value provide for zipcode")
    if (len(arg_zip) > 2) or (arg_zip[1] not in valid_zips):
        raise Exception("bad value for zipcode")

    return int(arg_zip[1])

zipcode = -1
try:
    zipcode = parse_zipcode()
except Exception as error:
    print("Error: "+str(error))
    print("Usage: python mono_log.py -zipcode=x")

if (zipcode != -1):


