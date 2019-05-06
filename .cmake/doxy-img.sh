#!/bin/sh

main () {
    output_dir="$1"
    readme="$2"
    link_readme="${readme}.link"
    $(cat "${readme}" | grep -E -o "https.*/badges/.*.svg" > "${link_readme}")
    
    while read link; do 
        img=$(echo "$link" | grep -i -E -o "badges.*.svg" | perl -pe 's/\//\./gsm')
        curl -sS "$link" > "${output_dir}/${img}"
        echo "Download $link to $img"; 
    done < "${link_readme}"
    
    rm "${link_readme}"

    perl -pe 's/https.*(badges.*\.svg)/img\/$1/gsm' -i "${readme}"
    perl -pe 's/img\/(.*)\/(.*)\/(.*)\.svg/img\/$1.$2.$3.svg/gsm' -i "${readme}"
}

main "$@"
