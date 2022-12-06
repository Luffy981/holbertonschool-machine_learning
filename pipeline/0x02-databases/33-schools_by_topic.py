#!/usr/bin/env python3
"""
Function that returns list of school having specific topics
"""


def schools_by_topic(mongo_collection, topic):
    """
    Returns a list of school having specific topics
    """
    return mongo_collection.find({"topics": {"$in": [topic]}})
