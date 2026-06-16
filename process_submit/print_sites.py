#!/usr/bin/env python
# Print out site names
import ausLib
import argparse
parser = argparse.ArgumentParser(description='Print out site names')
parser.add_argument("--one_line",action='store_true',
                    help='Print out site names, space seperated, all on one line')
parser.add_argument("--remove",nargs='*',help='Names to remove from list',default=[])

args= parser.parse_args()

if args.one_line:
    end=" "
else:
    end=None

remove = [r.lower() for r in args.remove]

            
for site in ausLib.site_numbers.keys():
    if site.lower() in remove:
        continue
    print(site,end=end)

if end is not None:
    print()
