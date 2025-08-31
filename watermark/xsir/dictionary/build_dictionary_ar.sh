set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python3 $SCRIPT_DIR/build_dictionary.py \
    --dicts \
        $SCRIPT_DIR/download/ar-en.txt \
        $SCRIPT_DIR/download/en-ar.txt \
    --output_file $SCRIPT_DIR/dictionary_ar.txt \
    --append_meta_symbols

python3 $SCRIPT_DIR/build_dictionary.py \
    --dicts \
        $SCRIPT_DIR/download/ar-en.txt \
        $SCRIPT_DIR/download/en-ar.txt \
    --output_file $SCRIPT_DIR/dictionary_no_meta_ar.txt