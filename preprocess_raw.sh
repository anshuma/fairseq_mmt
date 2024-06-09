src='en'
tgt='de'

#TEXT=data/multi30k-en-$tgt
TEXT=final_raw_data

fairseq-preprocess --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test.2016\
  --destdir raw_data-bin/multi30k.en-$tgt \
  --workers 8 --joined-dictionary 
