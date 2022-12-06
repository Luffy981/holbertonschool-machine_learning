#!/usr/bin/env python3
"""
Function that inserts documents in Python
"""


def insert_school(mongo_collection, **kwargs):
    """
    Inserts documents
    """
    return mongo_collection.insert_one(kwargs).inserted_id
