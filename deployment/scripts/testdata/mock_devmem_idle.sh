#!/bin/sh
[ "$#" -eq 2 ] || exit 2
[ "$1" = 0xA0000000 ] || exit 2
[ "$2" = 32 ] || exit 2
echo 0x00000004

