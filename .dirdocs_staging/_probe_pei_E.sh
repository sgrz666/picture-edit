#!/bin/bash
R=/data/shangguanrz/pic-edit-experiments
echo '===== TOP ls -la ====='
ls -la $R
echo '===== TOP direct files: identify ====='
find $R -maxdepth 1 -type f | while read f; do echo "-- $f"; file "$f"; stat -c '%s bytes, mtime %y' "$f"; done
echo '===== each direct file content ====='
find $R -maxdepth 1 -type f -name '*.md' -o -maxdepth 1 -type f -name '*.txt' -o -maxdepth 1 -type f -name '*.json' | while read f; do echo "@@@@@ $f"; head -c 4000 "$f"; echo; done
echo '===== 8arm ls -la ====='
ls -la $R/pei2i_body_face_8arm_20261009
echo '===== 8arm direct files head ====='
find $R/pei2i_body_face_8arm_20261009 -maxdepth 1 -type f | while read f; do echo "@@@@@ $f ($(stat -c %s $f) bytes, mtime $(stat -c %y $f))"; head -c 6000 "$f"; echo; done
echo '===== 8arm du ====='
du -sh --si $R/pei2i_body_face_8arm_20261009/*
