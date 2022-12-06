#!/usr/bin/env python3
"""
Function that returns all students sorted by average score in MongoDB
"""


def top_students(mongo_collection):
    """
    Returns all students sorted by average score
    """
    students = []
    documents = mongo_collection.find()
    for student in documents:
        total_score = 0
        topics = student["topics"]
        for project in topics:
            total_score += project["score"]
        average_score = total_score / len(topics)
        student["averageScore"] = average_score
        students.append(student)
    students = sorted(students, key=lambda i: i["averageScore"], reverse=True)
    return students
