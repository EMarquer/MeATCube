"""Test the creation of a CB, in particular: 
- can it handle the various kinds of data as CB?
- can it handle the various kinds of data as input to the functions?
"""

# load meatcube
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import meatcube as mc

def test_creation_empty_cb():
    # Arrange
    cb = mc.CB([], [], [], lambda x,y: 0, lambda x,y: 0)
    # Act
    # Assert
    pass