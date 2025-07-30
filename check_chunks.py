#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('chunks.db')
cursor = conn.cursor()
cursor.execute('SELECT chunk_text FROM chunks WHERE document_name LIKE "%Performance%" LIMIT 2')
chunks = cursor.fetchall()
print('Sample chunks from Performance Analysis:')
for chunk in chunks:
    print(f'Chunk: {chunk[0][:200]}...')
    print('---')
conn.close()
