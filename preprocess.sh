src='en'
tgt='de'

#TEXT=data/multi30k-en-$tgt
TEXT=small_dataset/data/translated_multi30k-en-de
fairseq-preprocess --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test.2016,$TEXT/test.2017,$TEXT/test.coco \
  --destdir translated-data-bin/multi30k.en-$tgt \
  --workers 8 --joined-dictionary 
