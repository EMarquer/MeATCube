"""


echo datasets/*/ | tr + _ | tr ' ' '\n' | sed -r 's/datasets\/(.+?)\//preprocess\/\1.py/g' | xargs touch

"""
