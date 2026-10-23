#!/bin/bash
# Sources for post hoc PH5 (emp_provenance.py). Network only, no compute.
#  1. Pham et al. 2005 EMP release from the Internet Archive (jaist.ac.jp/~tran/nucleosome is offline).
#     The archive's CDX index (checked 2026-09-27) holds only the datasets page and dataset01/02.txt
#     (H3, H4 occupancy); dataset04-14 were never archived. emp_provenance.py uses whatever is present.
#  2. S. cerevisiae sacCer3 genome and the SGD ORF table (sgdGene) from UCSC.
# Usage: src/upgrades/gue_suite/fetch_emp_sources.sh   (from the repo root)
set -e
cd "$(dirname "$0")/../../.."
P=external/gue/pham2005; Y=external/gue/yeast_genome
mkdir -p $P $Y
curl -sL --max-time 120 -o $P/members_wayback_20070715.htm \
  "http://web.archive.org/web/20070715153138/http://www.jaist.ac.jp/~tran/nucleosome/members.htm"
for n in 01 02; do
  curl -sL --max-time 600 -o $P/dataset$n.txt \
    "http://web.archive.org/web/2019id_/http://www.jaist.ac.jp/~tran/nucleosome/ten_dataset/dataset$n.txt"
  head -c1 $P/dataset$n.txt | grep -q '>' || { echo "dataset$n.txt: not a FASTA file"; rm -f $P/dataset$n.txt; }
done
curl -s --max-time 60 "http://web.archive.org/cdx/search/cdx?url=jaist.ac.jp/~tran/nucleosome/ten_dataset/*&fl=timestamp,original,statuscode" > $P/cdx_listing.txt || true
curl -sL --max-time 300 -o $Y/sacCer3.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/sacCer3/bigZips/sacCer3.fa.gz
curl -sL --max-time 300 -o $Y/sgdGene.txt.gz https://hgdownload.soe.ucsc.edu/goldenPath/sacCer3/database/sgdGene.txt.gz
shasum -a 256 $P/members_wayback_20070715.htm $P/dataset*.txt $Y/sacCer3.fa.gz $Y/sgdGene.txt.gz
# SHA-256 used for the reported results (2026-09-27):
#  members_wayback_20070715.htm 693e09f1567397a2da3d9ce49299a7175baba936f05b736d35e9e6b831526ff7
#  dataset01.txt aea90363934be8a0cb197b6883ad66ae14174e862c3da9b172a36640b96074e4
#  dataset02.txt d9df61a55254bddb234083f7194e6a92d62f3edb236272e1a9ab5e46a2261eeb
#  sacCer3.fa.gz e3a70396853eff5a012077efbd06cb97d858e12349745d102e316bb8f620f266
#  sgdGene.txt.gz 52f8b74d483367dd835e3128e64631e8840f3eecfbe07504967953515d5e9f5f
