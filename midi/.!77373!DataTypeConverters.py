# -*- coding: ISO-8859-1 -*-

from struct import pack, unpack

"""
This module contains functions for reading and writing the special data types
that a midi file contains.
"""

"""
nibbles are four bits. A byte consists of two nibles.
hiBits==0xF0, loBits==0x0F Especially used for setting
channel and event in 1. byte of musical midi events
"""



def getNibbles(byte):
    """
    Returns hi and lo bits in a byte as a tuple
    >>> getNibbles(142)
    (8, 14)
    
    Asserts byte value in byte range
    >>> getNibbles(256)
    Traceback (most recent call last):
        ...
    ValueError: Byte value out of range 0-255: 256
    """
    if not 0 <= byte <= 255:
        raise ValueError('Byte value out of range 0-255: %s' % byte)
    return (byte >> 4 & 0xF, byte & 0xF)


def setNibbles(hiNibble, loNibble):
    """
    Returns byte with value set according to hi and lo bits
    Asserts hiNibble and loNibble in range(16)
    >>> setNibbles(8, 14)
    142
    
    >>> setNibbles(8, 16)
    Traceback (most recent call last):
        ...
    ValueError: Nible value out of range 0-15: (8, 16)
    """
    if not (0 <= hiNibble <= 15) or not (0 <= loNibble <= 15):
        raise ValueError('Nible value out of range 0-15: (%s, %s)' % (hiNibble, loNibble))
    return (hiNibble << 4) + loNibble



def readBew(value):
    """
    Reads string as big endian word, (asserts len(value) in [1,2,4])
