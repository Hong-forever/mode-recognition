#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python 3 format checker for submissions in: Team Run ReviewID polarity
- Ensures 2500 non-empty lines
- Each line has exactly 4 whitespace-separated columns
- Column 3 (ReviewID) is an integer >= 0 (no upper bound enforced)
- Column 4 (polarity) is either 'positive' or 'negative'

Usage:
    python check_format_py3.py <submission_file>
"""
import sys
import copy


def main():
    if len(sys.argv) != 2:
        print("Usage: python check_format_py3.py <submission_file>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        with open(path, 'r', encoding='utf-8') as reader:
            lines_1 = reader.read().split('\n')
    except UnicodeDecodeError:
        with open(path, 'r', encoding='gb18030', errors='ignore') as reader:
            lines_1 = reader.read().split('\n')

    lines = []
    for line in lines_1:
        line_temp = copy.deepcopy(line)
        if len(line_temp.strip()) != 0:
            lines.append(line)

    right = True
    if len(lines) != 2500:
        print('row count error: expected 2500, got', len(lines))
        right = False

    for i, line in enumerate(lines):
        str_list = line.split()
        if len(str_list) != 4:
            print(f'column count error at row {i}: expected 4, got {len(str_list)}')
            right = False
            continue

        # Column 1: Team (string) – no strict check
        # Column 2: Run (string/integer) – no strict check

        # Column 3: ReviewID integer >= 0
        try:
            sample_id = int(str_list[2])
            if sample_id < 0:
                print(f'id number error at row {i}: ReviewID must be >= 0')
                right = False
        except Exception:
            print(f'3rd column at row {i} must be integer')
            right = False

        # Column 4: polarity
        if str_list[3] not in ('negative', 'positive'):
            print(f'4th column at row {i} must be positive or negative')
            right = False

    if right:
        print('check passed...')
    else:
        sys.exit(2)


if __name__ == '__main__':
    main()
