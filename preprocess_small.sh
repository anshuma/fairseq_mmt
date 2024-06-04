src='en'
tgt='de'

TEXT=small_dataset/data/multi30k-en-$tgt

fairseq-preprocess --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test.2016,$TEXT/test.2017,$TEXT/test.coco \
  --destdir small_dataset/data-bin/multi30k.en-$tgt \
  --workers 8 --joined-dictionary 
