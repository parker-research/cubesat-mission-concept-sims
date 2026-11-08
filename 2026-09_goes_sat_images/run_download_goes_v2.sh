#!/bin/bash

SCRIPT_PATH=.../download_goes_v2.py
OUT_PATH=.../goes_images

uv run $SCRIPT_PATH --out $OUT_PATH/goes18_geocolor_10848x10848/  --url 'https://cdn.star.nesdis.noaa.gov/GOES18/ABI/FD/GEOCOLOR/' --resolution 10848x10848
uv run $SCRIPT_PATH --out $OUT_PATH/goes18_geocolor_21696x21696/  --url 'https://cdn.star.nesdis.noaa.gov/GOES18/ABI/FD/GEOCOLOR/' --resolution 21696x21696
uv run $SCRIPT_PATH --out $OUT_PATH/goes19_geocolor_10848x10848/  --url 'https://cdn.star.nesdis.noaa.gov/GOES19/ABI/FD/GEOCOLOR/' --resolution 10848x10848
uv run $SCRIPT_PATH --out $OUT_PATH/goes19_geocolor_21696x21696/  --url 'https://cdn.star.nesdis.noaa.gov/GOES19/ABI/FD/GEOCOLOR/' --resolution 21696x21696
