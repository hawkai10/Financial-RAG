#!/usr/bin/env python3
import sqlite3

# Find first year rent chunk
conn = sqlite3.connect('chunks.db')
cursor = conn.cursor()
cursor.execute('SELECT chunk_id, SUBSTR(chunk_text, 1, 300) FROM chunks WHERE chunk_text LIKE "%40000%" AND document_name LIKE "%rent%"')
results = cursor.fetchall()
print('=== First year rent chunks (40000) ===')
for r in results:
    print(f'ID: {r[0]}')
    print(f'Text: {r[1]}...')
    print('---')

# Find second year rent chunk
cursor.execute('SELECT chunk_id, SUBSTR(chunk_text, 1, 300) FROM chunks WHERE chunk_text LIKE "%42800%" AND document_name LIKE "%rent%"')
results = cursor.fetchall()
print('\n=== Second year rent chunks (42800) ===')
for r in results:
    print(f'ID: {r[0]}')
    print(f'Text: {r[1]}...')
    print('---')
    
conn.close()
