"""Import the pinned KiCad Jetson example into a separate analysis directory."""
import argparse
import hashlib
import json
from pathlib import Path

from kicad2ems.board import Board
from kicad2ems.importers import open_board


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    raw=args.board.read_bytes()
    sha=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
    if sha!='2884cd5276b0676d4e5d8d919f202d85e8432987':
        parser.error('board differs from the pinned example used in this report')
    args.out.mkdir(parents=True,exist_ok=True)
    board=Board(open_board(str(args.board)),need_ports=False,parts='none',log=print).to_dict()
    (args.out/'board.json').write_text(json.dumps(board))
    print('Verified source blob and wrote',args.out/'board.json')


if __name__=='__main__':main()
