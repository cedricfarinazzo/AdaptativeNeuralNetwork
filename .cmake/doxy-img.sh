#!/bin/sh

main () {
    html_file="$1"
    echo "Update svg image link in ${html_file}"
    perl -pe 's/<object.*data="(.+)".*alt="(.*)">/<img src="$1" alt="$2" \/>/gsm' -i "${html_file}"
}

main "$@"
