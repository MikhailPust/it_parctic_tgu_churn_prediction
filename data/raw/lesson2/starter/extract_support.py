"""Starter extraction only. It does not clean or combine the sources."""
import argparse
import csv
import sqlite3
from pathlib import Path

QUERY = '''SELECT ticket_id, customer_id, created_at, available_at,
                  topic, channel, resolved_at, resolution_available_at,
                  satisfaction_score
           FROM support_current_raw'''

def extract(db,output):
    output.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(f'file:{db.resolve()}?mode=ro',uri=True) as conn:
        cursor=conn.execute(QUERY)
        columns=[item[0] for item in cursor.description]
        rows=cursor.fetchall()
    with output.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f,delimiter=';',lineterminator='\n');w.writerow(columns);w.writerows(rows)
    print(f'Extracted {len(rows)} rows into {output}')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--db',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();extract(args.db,args.output)
